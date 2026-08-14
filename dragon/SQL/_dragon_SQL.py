import sqlite3
import pandas as pd
from pathlib import Path
from typing import Union, Any, Optional, Literal

from .._core import get_logger
from ..path_manager import make_fullpath, sanitize_filename


_LOGGER = get_logger("DragonSQL")


__all__ = [
    "DragonSQL",
]


class DragonSQL:
    """
    A user-friendly context manager for handling SQLite database operations.

    This class abstracts the underlying sqlite3 connection and cursor management,
    providing simple methods to execute queries, create tables, and handle data
    insertion and retrieval using pandas DataFrames.

    Parameters
    ----------
    db_path : Union[str, Path]
        The file path to the SQLite database. If the file does not exist,
        it will be created upon connection.

    Example
    -------
    >>> schema = {
    ...     "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    ...     "run_name": "TEXT NOT NULL",
    ...     "feature_a": "REAL",
    ...     "score": "REAL"
    ... }
    >>> with DragonSQL("my_results.db") as db:
    ...     db.create_table("experiments", schema)
    ...     data = {"run_name": "first_run", "feature_a": 0.123, "score": 95.5}
    ...     db.insert_row("experiments", data)
    ...     df = db.query_to_dataframe("SELECT * FROM experiments")
    ...     print(df)
    """
    def __init__(self, db_path: Union[str, Path]):
        """Initializes the DragonSQL with the path to the database file."""
        if isinstance(db_path, str):
            if not db_path.endswith(".db"):
                db_path = db_path + ".db"
        elif isinstance(db_path, Path):
            if db_path.suffix != ".db":
                db_path = db_path.with_suffix(".db")
        
        self.db_path = make_fullpath(db_path, make=True, enforce="file")
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def __enter__(self):
        """Establishes the database connection and returns the manager instance."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            _LOGGER.info(f"❇️ Successfully connected to database: {self.db_path}")
            return self
        except sqlite3.Error as e:
            _LOGGER.error(f"Database connection failed: {e}")
            raise  # Re-raise the exception after logging

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Commits changes and closes the database connection."""
        if self.conn:
            if exc_type:  # If an exception occurred, rollback
                self.conn.rollback()
                _LOGGER.warning("Rolling back transaction due to an error.")
            else:  # Otherwise, commit the transaction
                self.conn.commit()
            self.conn.close()
            _LOGGER.info(f"Database connection closed: {self.db_path.name}")

    def create_table(self, table_name: str, schema: dict[str, str], if_not_exists: bool = True):
        """
        Creates a new table in the database based on a provided schema.

        Parameters
        ----------
        table_name : str
            The name of the table to create.
        schema : Dict[str, str]
            A dictionary where keys are column names and values are their SQL data types
            (e.g., {"id": "INTEGER PRIMARY KEY", "name": "TEXT NOT NULL"}).
        if_not_exists : bool, default=True
            If True, adds "IF NOT EXISTS" to the SQL statement to prevent errors
            if the table already exists.
        """
        if not self.cursor:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
        
        sanitized_table_name = sanitize_filename(table_name)

        columns_def = ", ".join([f'"{col_name}" {col_type}' for col_name, col_type in schema.items()])
        exists_clause = "IF NOT EXISTS" if if_not_exists else ""
        
        query = f"CREATE TABLE {exists_clause} {sanitized_table_name} ({columns_def})"
        
        _LOGGER.info(f"➡️ Executing: {query}")
        self.cursor.execute(query)

    def insert_row(self, table_name: str, data: dict[str, Any]):
        """
        Inserts a single row of data into the specified table.

        Parameters
        ----------
        table_name : str
            The name of the target table.
        data : Dict[str, Any]
            A dictionary where keys correspond to column names and values are the
            data to be inserted.
        """
        if not self.cursor:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
        
        sanitized_table_name = sanitize_filename(table_name)

        columns = ', '.join(f'"{k}"' for k in data.keys())
        placeholders = ', '.join(['?'] * len(data))
        values = list(data.values())
        
        query = f'INSERT INTO "{sanitized_table_name}" ({columns}) VALUES ({placeholders})'
        
        self.cursor.execute(query, values)

    def query_to_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Executes a SELECT query and returns the results as a pandas DataFrame.

        Parameters
        ----------
        query : str
            The SQL SELECT statement to execute.
        params : Optional[tuple], default=None
            An optional tuple of parameters to pass to the query for safety
            against SQL injection.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the query results.
        """
        if not self.conn:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
            
        return pd.read_sql_query(query, self.conn, params=params)

    def execute_sql(self, query: str, params: Optional[tuple] = None):
        """
        Executes an arbitrary SQL command that does not return data (e.g., UPDATE, DELETE).

        Parameters
        ----------
        query : str
            The SQL statement to execute.
        params : Optional[tuple], default=None
            An optional tuple of parameters for the query.
        """
        if not self.cursor:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
        
        self.cursor.execute(query, params if params else ())

    def insert_many(self, table_name: str, data: list[dict[str, Any]]):
        """
        Inserts multiple rows into the specified table in a single, efficient transaction.

        Parameters
        ----------
        table_name : str
            The name of the target table.
        data : List[Dict[str, Any]]
            A list of dictionaries, where each dictionary represents a row to be inserted.
            All dictionaries should have the same keys.
        """
        if not self.cursor:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
        if not data:
            _LOGGER.warning("'insert_many' called with empty data list. No action taken.")
            return
        
        sanitized_table_name = sanitize_filename(table_name)

        # Assume all dicts have the same keys as the first one
        first_row = data[0]
        columns = ', '.join(f'"{k}"' for k in first_row.keys())
        placeholders = ', '.join(['?'] * len(first_row))
        
        # Create a list of tuples, where each tuple is a row of values
        values_to_insert = [list(row.values()) for row in data]

        query = f'INSERT INTO "{sanitized_table_name}" ({columns}) VALUES ({placeholders})'
        
        self.cursor.executemany(query, values_to_insert)
        _LOGGER.info(f"➡️ Bulk inserted {len(values_to_insert)} rows into '{sanitized_table_name}'.")
        
    def insert_from_dataframe(self, table_name: str, df: pd.DataFrame, if_exists: Literal['fail', 'replace', 'append'] = 'append'):
        """
        Writes records from a pandas DataFrame to the specified SQL table.

        Parameters
        ----------
        table_name : str
            The name of the target SQL table.
        df : pd.DataFrame
            The DataFrame to be written.
        if_exists : str, default 'append'
            How to behave if the table already exists.
            - 'fail': Raise a ValueError.
            - 'replace': Drop the table before inserting new values.
            - 'append': Insert new values to the existing table.
        """
        if not self.conn:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
        
        sanitized_table_name = sanitize_filename(table_name)

        df.to_sql(
            sanitized_table_name,
            self.conn,
            if_exists=if_exists,
            index=False  # Typically, we don't want to save the DataFrame index
        )
        _LOGGER.info(f"➡️ Wrote {len(df)} rows from DataFrame to table '{table_name}' using mode '{if_exists}'.")
        
    def list_tables(self) -> list[str]:
        """Returns a list of all table names in the database."""
        if not self.cursor:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
        
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # The result of the fetch is a list of tuples, e.g., [('table1',), ('table2',)]
        return [table[0] for table in self.cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """
        Retrieves the schema of a specific table and returns it as a DataFrame.

        Returns a DataFrame with columns: cid, name, type, notnull, dflt_value, pk
        """
        if not self.conn:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
        
        sanitized_table_name = sanitize_filename(table_name)
            
        # PRAGMA is a special SQL command in SQLite for database metadata
        return pd.read_sql_query(f'PRAGMA table_info("{sanitized_table_name}");', self.conn)

    def create_index(self, table_name: str, column_name: str, unique: bool = False):
        """
        Creates an index on a column of a specified table to speed up queries.

        Parameters
        ----------
        table_name : str
            The name of the table containing the column.
        column_name : str
            The name of the column to be indexed.
        unique : bool, default=False
            If True, creates a unique index, which ensures all values in the
            column are unique.
        """
        if not self.cursor:
            _LOGGER.error("Database connection is not open.")
            raise sqlite3.Error()
        
        sanitized_table_name = sanitize_filename(table_name)

        index_name = f"idx_{sanitized_table_name}_{column_name}"
        unique_clause = "UNIQUE" if unique else ""
        
        query = f'CREATE {unique_clause} INDEX IF NOT EXISTS "{index_name}" ON "{sanitized_table_name}" ("{column_name}")'
        
        _LOGGER.info(f"➡️ Executing: {query}")
        self.cursor.execute(query)
        
    def commit(self):
        """Manually commits the current transaction."""
        if self.conn:
            self.conn.commit()
            _LOGGER.debug(f"Transaction committed to {self.db_path.name}")
        else:
            _LOGGER.error("Cannot commit: Database connection is not open.")

