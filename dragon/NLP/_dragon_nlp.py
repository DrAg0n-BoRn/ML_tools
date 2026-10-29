from numbers import Integral
import numpy as np
import pandas as pd
import polars as pl
from typing import Union, Optional, Literal, Any
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from ..serde import serialize_object_filename, deserialize_object

from ..IO import save_json, save_list_strings
from .._core import get_logger


_LOGGER = get_logger("NLP Embedder")


# Main dependency
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _LOGGER.error("NLP dependencies missing. Install them using the extra 'dragon-ml-toolbox[nlp]'.")
    raise ImportError()


__all__ = [
    "DragonNLP"
]


class DragonNLP:
    """
    Transforms raw text columns into tabular numerical format suitable for machine learning.
    
    This pipeline performs:
    1. Statistical Feature Extraction (lengths, word counts, punctuation).
    2. TF-IDF (Bag-of-Words) vectorization.
    3. TruncatedSVD (Lexical exact-match features).
    4. Dense NLP Embeddings (Semantic context via sentence-transformers).
    
    ### Note:
    
    - Output contains only generated numerical features and preserves pandas indexes. 
    - Input columns and feature order are learned once per successful fit.
    - Additional inference columns are ignored; missing required columns fail. 
    - A failed refit leaves the previously fitted artifacts intact.
    
    """
    def __init__(self, *,
                 text_columns: Union[list[str], tuple[str, ...], None],
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 svd_components: Optional[int] = 30,
                 strategy: Literal["concat", "independent"] = "concat",
                 batch_size: int = 32,
                 scale_statistics: bool = True,
                 random_state: Optional[int] = 42,
                 empty_vocabulary: Literal["skip", "raise"] = "raise",
                 max_features: Optional[int] = 10000,
                 tfidf_dtype: Literal["float32", "float64"] = "float32"):
        """
        Initializes the NLP Vectorizer pipeline.
        
        Args:
            text_columns (list[str] | tuple[str, ...] | None): Explicit ordered columns, or None to discover all String columns during fitting.
            embedding_model_name (str): The HuggingFace model string. 
                - 'all-MiniLM-L6-v2' A small, fast model with 384 dimensions, ~80 MB
                - 'all-mpnet-base-v2' The gold standard for medium-sized models with 768 dimensions, ~420 MB
                - 'BAAI/bge-large-en-v1.5' A large, high-quality model with 1024 dimensions, ~1.4 GB. Requires GPU for speed.
                - 'hkunlp/instructor-xl' A massive, high-quality model with 768 dimensions, ~5.0 GB. Requires GPU.
            svd_components (int | None): Number of SVD dimensions to compress the TF-IDF matrix into.
            strategy (str): 'concat' to join all text columns into one before processing, or 'independent' to process each column separately.
            batch_size (int): Batch size for generating dense embeddings.
            scale_statistics (bool): Whether to standardize the statistical features.
            random_state (int | None): Random state for reproducible results.
            empty_vocabulary (str): 'skip' to ignore empty vocabulary, 'raise' to raise an error.
            max_features (int | None): Optional positive TF-IDF vocabulary limit.
            tfidf_dtype (str): Data type for TF-IDF features. float32 is faster and uses less memory; float64 is more precise.
            
        ### Note:
        
        - SVD is highly recommended for large vocabularies to reduce dimensionality and improve performance.
        - The 'concat' strategy is useful for short text columns, while 'independent' is better for semantically distinct columns. However, 
        'independent' will produce more features and require more memory.
        """
        # Validate the original argument before converting it.
        if text_columns is not None:
            self._validate_columns(text_columns)

        self._text_columns = None if text_columns is None else tuple(text_columns)
        self.embedding_model_name = embedding_model_name
        self.svd_components = svd_components
        self.strategy = strategy
        self.batch_size = batch_size
        self._scale_statistics = scale_statistics
        self.random_state = random_state
        self._empty_vocabulary = empty_vocabulary
        self._max_features = max_features
        self.tfidf_dtype = tfidf_dtype
        self._validate_config()

        self._tf_idf_vectorizers = {}
        self._svd_transformers = {}
        self._statistics_scaler = None
        self._embedding_model = None
        self._initialized_embedding_model_name = None
        self._fitted_state = None
        self._is_fitted = False
    
    @staticmethod
    def _positive_integer(name, value, allow_none=False):
        if allow_none and value is None:
            return
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            suffix = " or None" if allow_none else ""
            _LOGGER.error(f"{name} must be a positive integer{suffix}. Got: {value}")
            raise ValueError()

    @staticmethod
    def _validate_columns(columns):
        if not isinstance(columns, (list, tuple)):
            _LOGGER.error("text_columns must be a list or tuple of names.")
            raise ValueError()
        if not columns or any(not isinstance(c, str) or not c for c in columns):
            _LOGGER.error("text_columns must contain at least one nonempty string name.")
            raise ValueError()
        if len(set(columns)) != len(columns):
            _LOGGER.error("text_columns must contain unique names.")
            raise ValueError()

    def _validate_config(self):
        if self._text_columns is not None:
            self._validate_columns(self._text_columns)
        if not isinstance(self.embedding_model_name, str) or not self.embedding_model_name.strip():
            _LOGGER.error("embedding_model_name must be a nonempty string.")
            raise ValueError()
        if self.strategy not in ("concat", "independent"):
            _LOGGER.error("strategy must be 'concat' or 'independent'.")
            raise ValueError()
        if self._empty_vocabulary not in ("skip", "raise"):
            _LOGGER.error("empty_vocabulary must be 'skip' or 'raise'.")
            raise ValueError()
        if self.tfidf_dtype not in ("float32", "float64"):
            _LOGGER.error("tfidf_dtype must be 'float32' or 'float64'.")
            raise ValueError()
        self._positive_integer("svd_components", self.svd_components, allow_none=True)
        self._positive_integer("batch_size", self.batch_size)
        self._positive_integer("max_features", self._max_features, allow_none=True)
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state < 2**32
        ):
            _LOGGER.error("random_state must be None or an integer in [0, 2**32).")
            raise ValueError()
        if not isinstance(self._scale_statistics, bool):
            _LOGGER.error("scale_statistics must be a boolean.")
            raise ValueError()
    
    def _config(self):
        return {
            "text_columns": list(self._text_columns) if self._text_columns is not None else None,
            "embedding_model_name": self.embedding_model_name,
            "svd_components": None if self.svd_components is None else int(self.svd_components),
            "strategy": self.strategy,
            "batch_size": int(self.batch_size),
            "scale_statistics": self._scale_statistics,
            "random_state": None if self.random_state is None else int(self.random_state),
            "empty_vocabulary": self._empty_vocabulary,
            "max_features": None if self._max_features is None else int(self._max_features),
            "tfidf_dtype": self.tfidf_dtype
        }
    
    def initialize_embedding_model(self, 
                                   device: str = "cpu") -> None:
        """
        Loads the SentenceTransformer embedding model onto the specified device.
        
        Args:
            device (str): Device to load the embedding model on ('cpu', 'cuda', 'mps').
        """
        # initialize the embedding model
        _LOGGER.info(f"Loading SentenceTransformer model: {self.embedding_model_name} on {device}")
        try:
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
        except Exception as e:
            _LOGGER.error(f"Error loading SentenceTransformer model: {e}")
            raise
        else:
            self._initialized_embedding_model_name = self.embedding_model_name
            
    def _require_embedding_model(self, expected_name=None):
        if self._embedding_model is None:
            _LOGGER.error(
                "Embedding model is not initialized. Call initialize_embedding_model() before fitting or transforming nonempty data."
            )
            raise RuntimeError()
        if self._initialized_embedding_model_name not in (None, self.embedding_model_name):
            _LOGGER.error(
                "Embedding model name changed after initialization. Initialize the configured model before fitting."
            )
            raise ValueError()
        if expected_name is not None and self.embedding_model_name != expected_name:
            _LOGGER.error("Embedding model name changed after fitting. Refit before transform().")
            raise ValueError()
    
    def _require_fitted(self):
        if not self._is_fitted or self._fitted_state is None:
            _LOGGER.error("DragonNLP is not fitted. Call fit_transform() first.")
            raise RuntimeError()
        return self._fitted_state
    
    def _to_polars_and_texts(self, df, *, columns=None, strategy=None):
        """Select and normalize only the requested fields, in their fixed order."""
        if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            _LOGGER.error("df must be a pandas or Polars DataFrame.")
            raise TypeError()
        
        is_pandas = isinstance(df, pd.DataFrame)
        
        if is_pandas and not df.columns.is_unique:
            _LOGGER.error("Input pandas DataFrame column names must be unique.")
            raise ValueError()
        
        columns = self._text_columns if columns is None else tuple(columns)
        
        if is_pandas and columns is None and any(not isinstance(c, str) for c in df.columns):
            _LOGGER.error("Automatic text-column selection requires string column names.")
            raise TypeError()
        
        if columns is not None:
            self._validate_columns(columns)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                _LOGGER.error(f"Missing required text columns: {missing}.")
                raise ValueError()
            
        # Avoid converting unrelated pandas object columns during inference.
        selected = df.loc[:, list(columns)] if is_pandas and columns is not None else df
        pl_df: pl.DataFrame = pl.from_pandas(selected) if is_pandas else selected # type: ignore
        if columns is None:
            columns = tuple(c for c, dtype in pl_df.schema.items() if dtype == pl.String)
            if not columns:
                _LOGGER.error("No string columns found in the DataFrame. Provide text_columns explicitly.")
                raise ValueError()
            
        self._validate_columns(columns)
        normalized = {}
        for name in columns:
            # Some Polars Series methods internally use the Series name as an
            # expression selector. A safe temporary name prevents regex names
            # such as '^text$' from being interpreted as selectors.
            values = pl_df.get_column(name).alias("__dragon_text")
            string_type = values.dtype.base_type() in (pl.String, pl.Null, pl.Categorical, pl.Enum)
            if not string_type and values.null_count() != len(values):
                _LOGGER.error("Text column {name!r} has dtype {values.dtype}; convert non-null values to strings explicitly.")
                raise TypeError()
            normalized[name] = values.cast(pl.String).fill_null("").alias(name)
            
        text_df: pl.DataFrame = pl.DataFrame(normalized)
        strategy = self.strategy if strategy is None else strategy
        if strategy == "concat":
            texts = {
                "concat_text": text_df.select(
                    pl.concat_str(pl.all(), separator=" ").alias("text")
                ).get_column("text").to_list()
            }
        else:
            texts = {c: text_df.get_column(c).to_list() for c in columns}
        return is_pandas, text_df, texts
    
    def _extract_statistical_features(self, text_df: pl.DataFrame) -> pl.DataFrame:
        """
        Internal helper to calculate fast statistical features (character counts, 
        word counts, punctuation ratios, etc.) for the target text columns.
        """
        features = []
        for name in text_df.columns:
            values = text_df.get_column(name).alias("__dragon_text")
            features.extend([
                values.str.len_chars().alias(f"{name}_char_count"),
                values.str.count_matches(r"\w+").alias(f"{name}_word_count"),
                values.str.count_matches(r"[^\w\s]").alias(f"{name}_punct_count"),
            ])
        return pl.DataFrame(features)
    
    @staticmethod
    def _scale_statistical_features(statistics, scaler, *, fit=False):
        """Scale only the statistics block, preserving names and column order.

        A fresh scaler is fitted locally during fit_transform. Inference always
        reuses the committed scaler; its result is independent of batch size or
        the other rows in an inference batch. None preserves raw integer counts.
        """
        if scaler is None:
            return statistics

        values = statistics.to_numpy().astype(np.float64, copy=False)
        values = (
            scaler.fit_transform(values)
            if fit
            else scaler.transform(values)
        )

        if not np.isfinite(values).all():
            _LOGGER.error("Statistics scaling produced non-finite values.")
            raise ValueError()

        return pl.DataFrame(
            values,
            schema=statistics.columns,
            orient="row",
        )
    
    def _generate_embeddings(self, col_name, text_list, verbose, expected_dim=None):
        self._require_embedding_model()
        if verbose > 2:
            _LOGGER.info(f"Generating dense embeddings for: {col_name}")
            
        embeddings = np.asarray(self._embedding_model.encode( # type: ignore
            text_list,
            batch_size=int(self.batch_size),
            show_progress_bar=verbose > 1,
            convert_to_numpy=True,
        ))
        
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(text_list)
            or embeddings.shape[1] == 0
            ):
            _LOGGER.error(f"Encoder returned an invalid embedding shape for {col_name!r}: {embeddings.shape}.")
            raise ValueError()
        
        if expected_dim is not None and embeddings.shape[1] != expected_dim:
            _LOGGER.error(
                f"Embedding dimension changed for {col_name!r}: expected {expected_dim}, got {embeddings.shape[1]}. Refit the model."
            )
            raise ValueError()
        
        if embeddings.dtype.kind not in "fiu":
            _LOGGER.error(f"Encoder embeddings must be numeric. Got dtype: {embeddings.dtype}.")
            raise TypeError()
        
        embeddings = embeddings.astype(np.float32, copy=False)
        
        if not np.isfinite(embeddings).all():
            _LOGGER.error(f"Encoder embeddings contain non-finite values for {col_name!r}.")
            raise ValueError()
        
        names = [f"{col_name}_emb_{i}" for i in range(embeddings.shape[1])]
        
        return pl.DataFrame(embeddings, schema=names, orient="row")
    
    @staticmethod
    def _unique_feature_names(names):
        # Reserve natural names before choosing suffixes for duplicate names.
        reserved, used, result = set(names), set(), []
        for name in names:
            candidate, suffix = name, 2
            if candidate in used:
                while True:
                    candidate = f"{name}__{suffix}"
                    suffix += 1
                    if candidate not in reserved and candidate not in used:
                        break
            used.add(candidate)
            result.append(candidate)
        return tuple(result)
    
    @classmethod
    def _assemble_features(cls, blocks, height, fitted_schema=None):
        if any(block.height != height for block in blocks):
            _LOGGER.error("Feature blocks do not preserve the input row count.")
            raise ValueError()
        
        raw_names = tuple(name for block in blocks for name in block.columns)
        if fitted_schema is None:
            names = cls._unique_feature_names(raw_names)
        else:
            if raw_names != tuple(fitted_schema["raw_feature_names"]):
                _LOGGER.error("Generated features no longer match the fitted schema; refit.")
                raise ValueError()
            names = tuple(fitted_schema["feature_names"])
            
        renamed, offset = [], 0
        for block in blocks:
            updated = block.clone()
            updated.columns = list(names[offset:offset + block.width])
            renamed.append(updated)
            offset += block.width
            
        result = pl.concat(renamed, how="horizontal")
        
        if fitted_schema is not None and dict(result.schema) != fitted_schema["dtypes"]:
            _LOGGER.error("Generated feature dtypes no longer match the fitted schema; refit.")
            raise ValueError()
        return result, raw_names
    
    @staticmethod
    def _restore_output(final_df, original, is_pandas):
        if is_pandas:
            result = final_df.to_pandas()
            result.index = original.index.copy()
            return result
        return final_df
    
    def _commit_fitted_state(self, state):
        self._tf_idf_vectorizers = state["artifacts"]["tf_idf_vectorizers"]
        self._svd_transformers = state["artifacts"]["svd_transformers"]
        self._statistics_scaler = state["artifacts"]["statistics_scaler"]
        
        self.text_columns_ = tuple(state["schema"]["text_columns"])
        self.feature_names_out_ = tuple(state["schema"]["feature_names"])
        
        self._fitted_state = state
        self._is_fitted = True
        
    def _dense_tfidf_frame(self, matrix, vectorizer, col_name):
        names = [f"{col_name}_tfidf_{name}" for name in vectorizer.get_feature_names_out()]
        return pl.DataFrame(matrix.toarray(), schema=names, orient="row")

    def fit_transform(self, 
                      df: Union[pd.DataFrame, pl.DataFrame], 
                      verbose: int = 2) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Fits the TF-IDF and SVD transformers on the training data, generates the dense embeddings, 
        calculates statistical features, and concatenates them all into a final numerical DataFrame.
        
        Args:
            df (pd.DataFrame | pl.DataFrame): The raw training DataFrame containing the text columns.
            verbose (int): Verbosity level
                - 0: Silent
                - 1: Warnings
                - 2: Standard Info
                - 3: Detailed information
            
        Returns:
            df (pd.DataFrame | pl.DataFrame): A fully numerical DataFrame.
        """
        self._validate_config()
        self._require_embedding_model()
        
        is_pandas, text_df, texts = self._to_polars_and_texts(df)
        
        if text_df.height == 0:
            _LOGGER.error("Cannot fit DragonNLP on empty data.")
            raise ValueError()
        
        if verbose > 1:
            _LOGGER.info("Fitting text statistics, lexical features, and embeddings.")
            
        statistics_scaler = (
            StandardScaler()
            if self._scale_statistics
            else None
        )

        statistics = self._scale_statistical_features(
            self._extract_statistical_features(text_df),
            statistics_scaler,
            fit=True,
        )

        blocks = [statistics]
        
        vectorizers, svds, modes, embedding_dimensions = {}, {}, {}, {}
        for name, text_list in texts.items():
            vectorizer = TfidfVectorizer(
                max_features=self._max_features, dtype=np.dtype(self.tfidf_dtype).type
            )
            try:
                matrix = vectorizer.fit_transform(text_list)
            except ValueError as exc:
                if "empty vocabulary" not in str(exc).lower():
                    _LOGGER.error(f"TF-IDF fitting failed for {name!r}: {exc}")
                    raise ValueError() from exc
                if self._empty_vocabulary == "raise":
                    _LOGGER.error(f"TF-IDF for {name!r} has an empty vocabulary.")
                    raise ValueError() from exc
                modes[name] = "disabled"
                if verbose >= 1:
                    _LOGGER.warning(f"Skipping empty TF-IDF vocabulary for {name!r}.")
            else:
                vectorizers[name] = vectorizer
                if self.svd_components is not None and matrix.shape[1] >= 2:
                    n_components = min(
                        int(self.svd_components), matrix.shape[0], matrix.shape[1] - 1
                    )
                    svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
                    reduced = svd.fit_transform(matrix)
                    names = [f"{name}_svd_{i}" for i in range(reduced.shape[1])]
                    blocks.append(pl.DataFrame(reduced, schema=names, orient="row"))
                    svds[name], modes[name] = svd, "svd"
                else:
                    blocks.append(self._dense_tfidf_frame(matrix, vectorizer, name))
                    modes[name] = "tfidf"
                    
            embeddings = self._generate_embeddings(name, text_list, verbose)
            blocks.append(embeddings)
            embedding_dimensions[name] = embeddings.width
            
        final_df, raw_names = self._assemble_features(blocks, text_df.height)
        
        output = self._restore_output(final_df, df, is_pandas)
        
        state = {
            "config": self._config(),
            "artifacts": {
                "tf_idf_vectorizers": vectorizers,
                "svd_transformers": svds,
                "lexical_modes": modes,
                "embedding_dimensions": embedding_dimensions,
                "statistics_scaler": statistics_scaler,
            },
            "schema": {
                "text_columns": tuple(text_df.columns),
                "raw_feature_names": raw_names,
                "feature_names": tuple(final_df.columns),
                "dtypes": dict(final_df.schema),
            },
        }
        
        self._commit_fitted_state(state)
        
        return output

    def transform(self, 
                  df: Union[pd.DataFrame, pl.DataFrame], 
                  verbose: int = 2) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Transforms unseen test data or live inference data using the pre-fitted 
        TF-IDF and SVD artifacts, and generates dense embeddings.
        
        Args:
            df (pd.DataFrame | pl.DataFrame): Unseen DataFrame containing the text columns.
            verbose (int): Verbosity level
                - 0: Silent
                - 1: Warnings
                - 2: Standard Info
                - 3: Detailed information
        
        Returns:
            df (pd.DataFrame | pl.DataFrame): The processed numerical DataFrame.
        """
        state = self._require_fitted()
        self._positive_integer("batch_size", self.batch_size)
        
        # Fail early for nonempty input, before running any feature extraction.
        if isinstance(df, (pd.DataFrame, pl.DataFrame)) and len(df):
            self._require_embedding_model(state["config"]["embedding_model_name"])
        is_pandas, text_df, texts = self._to_polars_and_texts(
            df, columns=state["schema"]["text_columns"], strategy=state["config"]["strategy"]
        )
        if text_df.height == 0:
            empty = pl.DataFrame(schema=state["schema"]["dtypes"])
            return self._restore_output(empty, df, is_pandas)
        
        if verbose > 1:
            _LOGGER.info("Transforming text statistics, lexical features, and embeddings.")
            
        artifacts = state["artifacts"]
        statistics = self._scale_statistical_features(
            self._extract_statistical_features(text_df),
            artifacts["statistics_scaler"],
        )

        blocks = [statistics]
        
        for name, text_list in texts.items():
            mode = artifacts["lexical_modes"][name]
            if mode != "disabled":
                vectorizer = artifacts["tf_idf_vectorizers"][name]
                matrix = vectorizer.transform(text_list)
                if mode == "svd":
                    reduced = artifacts["svd_transformers"][name].transform(matrix)
                    names = [f"{name}_svd_{i}" for i in range(reduced.shape[1])]
                    blocks.append(pl.DataFrame(reduced, schema=names, orient="row"))
                else:
                    blocks.append(self._dense_tfidf_frame(matrix, vectorizer, name))
            blocks.append(self._generate_embeddings(
                name, text_list, verbose, expected_dim=artifacts["embedding_dimensions"][name]
            ))
            
        final_df, _ = self._assemble_features(blocks, text_df.height, state["schema"])
        
        return self._restore_output(final_df, df, is_pandas)
    
    def get_feature_names(self) -> np.ndarray:
        """Return a copy of the fitted numerical feature names, in output order as a NumPy array."""
        state = self._require_fitted()
        return np.asarray(state["schema"]["feature_names"], dtype=object).copy()

    def save(self, 
             directory: Union[str, Path], 
             filename: str = "dragon_nlp",
             verbose: bool = True) -> None:
        """
        Save fitted artifacts and schema, plus an informational JSON sidecar.

        The encoder weights are not included. Use the same model identifier or
        stable local model directory when reinitializing after loading.
        """
        state = self._require_fitted()
        
        if not isinstance(filename, str) or not filename.removesuffix(".joblib"):
            _LOGGER.error("filename must be a nonempty string.")
            raise ValueError()
        
        serialize_object_filename(
            obj=state, 
            save_dir=directory, 
            filename=filename,
            verbose=False, 
            raise_on_error=True,
        )
        
        info = {
            "description": "DragonNLP Serialized State",
            "embedding_model_name": state["config"]["embedding_model_name"],
            "fitted_columns": list(state["schema"]["text_columns"]),
            "feature_names": list(state["schema"]["feature_names"]),
            "lexical_modes": state["artifacts"]["lexical_modes"],
            "svd_components": state["config"]["svd_components"],
            "strategy": state["config"]["strategy"],
            "scale_statistics": state["config"]["scale_statistics"],
            "statistics_scaler": (
                "StandardScaler"
                if state["config"]["scale_statistics"]
                else None
            ),
        }
        
        save_json(
            data=info, 
            directory=directory,
            filename=filename.removesuffix(".joblib") + "_info.json", 
            verbose=False,
        )
        
        if verbose:
            _LOGGER.info(f"Saved DragonNLP state to '{directory}'.")
    
    @classmethod
    def _validate_saved_state(cls, state):
        """Reject incomplete state or a schema inconsistent with fitted artifacts."""
        
        if not isinstance(state, dict):
            _LOGGER.error("Invalid DragonNLP state: expected a dictionary.")
            raise ValueError()
        
        try:
            config, artifacts, schema = state["config"], state["artifacts"], state["schema"]
            if not all(isinstance(item, dict) for item in (config, artifacts, schema)):
                _LOGGER.error("Invalid DragonNLP state: config, artifacts, and schema must be dictionaries.")
                raise ValueError()
            
            cls._validate_columns(schema["text_columns"])
            if not isinstance(schema["feature_names"], (list, tuple)):
                _LOGGER.error("feature_names must be a list or tuple")
                raise ValueError()
            
            names = tuple(schema["feature_names"])
            if not names or len(set(names)) != len(names) or any(not isinstance(n, str) for n in names):
                _LOGGER.error("feature names must be nonempty and unique")
                raise ValueError()
            
            if (
                not isinstance(schema["raw_feature_names"], (list, tuple))
                or any(not isinstance(n, str) for n in schema["raw_feature_names"])
                or len(schema["raw_feature_names"]) != len(names)
            ):
                _LOGGER.error("raw feature names have the wrong length")
                raise ValueError()
            
            if tuple(schema["dtypes"]) != names:
                _LOGGER.error("feature dtypes names do not match the output order")
                raise ValueError()
            
            pl.DataFrame(schema=schema["dtypes"])
            
            # Validate scaler artifacts
            statistic_names = tuple(
                f"{column}_{suffix}"
                for column in schema["text_columns"]
                for suffix in ("char_count", "word_count", "punct_count")
            )
            statistic_width = len(statistic_names)

            if (
                tuple(schema["raw_feature_names"][:statistic_width]) != statistic_names
                or names[:statistic_width] != statistic_names
            ):
                _LOGGER.error("Statistics feature order does not match the text columns")
                raise ValueError()

            scaling_enabled = config["scale_statistics"]
            if not isinstance(scaling_enabled, bool):
                _LOGGER.error("scale_statistics must be a boolean")
                raise ValueError()

            scaler = artifacts["statistics_scaler"]

            if scaling_enabled:
                if (
                    not isinstance(scaler, StandardScaler)
                    or not scaler.with_mean # type: ignore
                    or not scaler.with_std # type: ignore
                    or getattr(scaler, "n_features_in_", None) != statistic_width
                ):
                    _LOGGER.error("missing or incompatible fitted statistics scaler")
                    raise ValueError()

                for attribute in ("mean_", "var_", "scale_"):
                    values = np.asarray(getattr(scaler, attribute))

                    if (
                        values.shape != (statistic_width,)
                        or not np.isfinite(values).all()
                    ):
                        _LOGGER.error(f"invalid statistics scaler attribute: {attribute}")
                        raise ValueError()

                if np.any(scaler.var_ < 0) or np.any(scaler.scale_ <= 0): # type: ignore
                    _LOGGER.error("statistics scaler variance/scale must be nonnegative/positive")
                    raise ValueError()

                sample_count = np.asarray(scaler.n_samples_seen_)
                if (
                    sample_count.ndim != 0
                    or not np.isfinite(sample_count)
                    or sample_count <= 0
                ):
                    _LOGGER.error("invalid statistics scaler training sample count")
                    raise ValueError()

                if any(
                    schema["dtypes"][name] != pl.Float64
                    for name in statistic_names
                ):
                    _LOGGER.error("scaled statistics must have Float64 output dtypes")
                    raise ValueError()

            elif scaler is not None:
                _LOGGER.error("scaler should be None when scaling is disabled")
                raise ValueError()
            
            groups = {"concat_text"} if config["strategy"] == "concat" else set(schema["text_columns"])
            modes, dimensions = artifacts["lexical_modes"], artifacts["embedding_dimensions"]
            vectorizers, svds = artifacts["tf_idf_vectorizers"], artifacts["svd_transformers"]
            if not all(isinstance(item, dict) for item in (modes, dimensions, vectorizers, svds)):
                _LOGGER.error("fitted artifacts must be dictionaries")
                raise ValueError()
            if set(modes) != groups or set(dimensions) != groups:
                _LOGGER.error("fitted text groups do not match the schema")
                raise ValueError()
            if any(mode not in ("disabled", "svd", "tfidf") for mode in modes.values()):
                _LOGGER.error("unknown lexical mode")
                raise ValueError()
            if set(vectorizers) != {name for name, mode in modes.items() if mode != "disabled"}:
                _LOGGER.error("missing or unexpected TF-IDF artifacts")
                raise ValueError()
            if set(svds) != {name for name, mode in modes.items() if mode == "svd"}:
                _LOGGER.error("missing or unexpected SVD artifacts")
                raise ValueError()
            for name in groups:
                cls._positive_integer("embedding dimension", dimensions[name])
                if name in vectorizers:
                    vectorizer = vectorizers[name]
                    if (
                        not isinstance(vectorizer, TfidfVectorizer)
                        or not hasattr(vectorizer, "vocabulary_")
                        or not hasattr(vectorizer, "idf_")
                    ):
                        _LOGGER.error("unfitted TF-IDF artifact")
                        raise ValueError()
                    vocabulary_size = len(vectorizer.vocabulary_)
                    idf = np.asarray(vectorizer.idf_)
                    if (
                        vocabulary_size == 0
                        or set(vectorizer.vocabulary_.values()) != set(range(vocabulary_size))
                        or idf.shape != (vocabulary_size,)
                        or not np.isfinite(idf).all()
                    ):
                        _LOGGER.error("inconsistent TF-IDF vocabulary or IDF weights")
                        raise ValueError()
                    
                if name in svds:
                    if not isinstance(svds[name], TruncatedSVD) or not hasattr(svds[name], "components_"):
                        _LOGGER.error("unfitted SVD artifact")
                        raise ValueError()
                    components = np.asarray(svds[name].components_)
                    if (
                        components.ndim != 2
                        or components.shape[0] == 0
                        or components.shape[1] != len(vectorizers[name].vocabulary_)
                        or not np.isfinite(components).all()
                    ):
                        _LOGGER.error("SVD and vocabulary dimensions disagree")
                        raise ValueError()
                    
            # Reconstruct the schema in the same order as fit_transform().
            ordered_groups = (
                ("concat_text",)
                if config["strategy"] == "concat"
                else tuple(schema["text_columns"])
            )
            expected_raw_names = list(statistic_names)
            expected_dtypes: list[Any] = [ pl.Float64 if scaling_enabled else pl.UInt32] * statistic_width
            float_dtypes = {
                np.dtype("float32"): pl.Float32,
                np.dtype("float64"): pl.Float64,
            }

            for name in ordered_groups:
                mode = modes[name]
                embedding_width = int(dimensions[name])
                if mode == "svd":
                    lexical_width = svds[name].components_.shape[0]
                elif mode == "tfidf":
                    lexical_width = len(vectorizers[name].vocabulary_)
                else:
                    lexical_width = 0

                # Reject oversized dimensions before allocating feature names.
                if len(expected_raw_names) + lexical_width + embedding_width > len(names):
                    _LOGGER.error(f"Feature widths exceed the saved schema for {name!r}.")
                    raise ValueError()

                if mode != "disabled":
                    lexical_dtype = np.dtype(vectorizers[name].dtype)
                    if mode == "svd":
                        lexical_dtype = np.result_type(
                            lexical_dtype, svds[name].components_.dtype
                        )
                        lexical_names = [f"{name}_svd_{i}" for i in range(lexical_width)]
                    else:
                        lexical_names = [
                            f"{name}_tfidf_{term}"
                            for term in vectorizers[name].get_feature_names_out()
                        ]

                    if lexical_dtype not in float_dtypes:
                        _LOGGER.error(f"Unsupported lexical dtype for {name!r}: {lexical_dtype}.")
                        raise ValueError()
                    expected_raw_names.extend(lexical_names)
                    expected_dtypes.extend([float_dtypes[lexical_dtype]] * lexical_width)

                expected_raw_names.extend(
                    f"{name}_emb_{i}" for i in range(embedding_width)
                )
                expected_dtypes.extend([pl.Float32] * embedding_width)

            if tuple(schema["raw_feature_names"]) != tuple(expected_raw_names):
                _LOGGER.error("Raw feature names or order do not match the fitted artifacts.")
                raise ValueError()

            expected_names = cls._unique_feature_names(expected_raw_names)
            if names != expected_names:
                _LOGGER.error("Feature names do not match the reconstructed unique names.")
                raise ValueError()

            expected_schema = dict(zip(expected_names, expected_dtypes))
            if not isinstance(schema["dtypes"], dict) or schema["dtypes"] != expected_schema:
                _LOGGER.error("Feature dtypes do not match the fitted artifacts.")
                raise ValueError()
                    
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            _LOGGER.error(f"Incomplete or invalid DragonNLP fitted state: {exc}")
            raise ValueError() from exc
    
    @classmethod
    def load(cls, 
             file_path: Union[str, Path],
             verbose: bool = True,
             *,
             initialize_embeddings: bool = False, 
             device: str = "cpu") -> 'DragonNLP':
        """
        Load fitted state; optionally initialize the encoder on device.
        
        Args:
            file_path (str | Path): Path to the serialized DragonNLP state.
            verbose (bool): If True, prints loading information.
            initialize_embeddings (bool): If True, initializes the SentenceTransformer model on the specified device.
            device (str): Device to load the embedding model on ('cpu', 'cuda', 'mps').
            
        Returns:
            DragonNLP: A DragonNLP instance with the fitted state loaded.
        """
        state = deserialize_object(
            filepath=file_path,
            expected_type=dict,
            verbose=False,
        )

        # Validate the format, schema, and fitted artifacts.
        cls._validate_saved_state(state)

        # Reconstruct the instance using the successful fit's configuration.
        try:
            instance = cls(**state["config"])
        except (TypeError, ValueError) as exc:
            _LOGGER.error(f"Failed to reconstruct DragonNLP instance from saved configuration: {exc}")
            raise ValueError() from exc

        # Restore artifacts, learned columns, output names, and fitted status.
        instance._commit_fitted_state(state)

        # Encoder weights are loaded separately from the serialized state.
        if initialize_embeddings:
            instance.initialize_embedding_model(device=device)

        if verbose:
            _LOGGER.info(f"Loaded DragonNLP state from '{file_path}'.")

        return instance
    
    def save_feature_names(
        self,
        directory: Union[str, Path],
        filename: str = "dragon_nlp_features",
        save_as: Literal["txt", "json"] = "txt",
        verbose: bool = True,
    ) -> None:
        """
        Saves all fitted feature names in output order to a text file or JSON file.
        
        Args:
            directory (str | Path): Directory to save the feature names file.
            filename (str): Name of the file to save the feature names.
            save_as ("txt" | "json"): Format to save the feature names in.
            verbose (bool): If True, prints a saving confirmation message.
        """
        names: list[str] = self.get_feature_names().tolist()
        
        if save_as not in ("txt", "json"):
            _LOGGER.error("save_as must be either 'txt' or 'json'.")
            raise ValueError()
        
        if save_as == "txt":
            if any("\n" in name or "\r" in name or name != name.strip() for name in names):
                _LOGGER.warning("Feature names containing line breaks or leading/trailing whitespace cannot be saved as a text file. Saving as JSON instead.")
                save_as = "json"
            else:
                save_list_strings(
                    list_strings=names,
                    directory=directory,
                    filename=filename,
                    verbose=False
                )
        
        # JSON default
        if save_as == "json":
            save_json(
                data=names,
                directory=directory,
                filename=filename,
                verbose=False,
            )
            
        if verbose:
            _LOGGER.info(f"Saved DragonNLP feature names to '{directory}'.")

