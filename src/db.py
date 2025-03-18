import psycopg2
import pandas as pd
from psycopg2 import sql
import os

# Database credentials (Update these values)
DB_CONFIG = {
    "dbname": "mnist_db",
    "user": "postgres",
    "password": "Password@23",  # Replace with your PostgreSQL password
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def log_prediction(predicted_digit, true_label=None):
    """Log model predictions into the PostgreSQL database."""
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO predictions (predicted_digit, true_label)
            VALUES (%s, %s)
        """)
        cursor.execute(insert_query, (predicted_digit, true_label))
        conn.commit()
        cursor.close()
        print("✅ Prediction logged successfully!")
    except Exception as e:
        print(f"❌ Error logging prediction: {e}")
    finally:
        conn.close()


def fetch_predictions():
    """Fetch logged predictions from the PostgreSQL database."""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        query = """
            SELECT timestamp, predicted_digit, true_label
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 10;
        """  # ✅ Query is now a string
        
        df = pd.read_sql_query(query, conn)  # Fetch as DataFrame
        conn.close()
        return df

    except Exception as e:
        print(f"❌ Error fetching predictions: {e}")
        return None
    