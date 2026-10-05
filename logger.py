try:
    import psycopg2
except ImportError:
    psycopg2 = None

from datetime import datetime
import os
import csv

# --- DATABASE CONFIGURATION ---
# Load from environment variables
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

def log_to_csv(timestamp, ip_address, user_agent, filename, file_ext, size_mb, target_format, status):
    try:
        csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, 'activity_log.csv')
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Timestamp", "IP Address", "User Device (Agent)", "File Name", "Original Extension", "File Size (MB)", "Target Format", "Status"])
            writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S'), ip_address, user_agent, filename, file_ext, size_mb, target_format, status])
    except Exception as csv_err:
        print(f"!!! CSV LOGGING ERROR: {csv_err}")

def log_upload_details(request, filename, file_size_bytes, target_format, status="Success"):
    """
    Logs upload details. Attempts PostgreSQL first if configured, falls back to CSV.
    """
    # 1. GATHER DATA
    timestamp = datetime.now()
    
    # Robust IP detection
    if request.headers.getlist("X-Forwarded-For"):
        ip_address = request.headers.getlist("X-Forwarded-For")[0]
    else:
        ip_address = request.remote_addr

    user_agent = request.headers.get('User-Agent')
    file_ext = os.path.splitext(filename)[1]
    size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Check if database driver is available and configured
    if not psycopg2 or not DB_NAME or not DB_PASSWORD:
        # Fall back directly to local CSV logging
        log_to_csv(timestamp, ip_address, user_agent, filename, file_ext, size_mb, target_format, status)
        return

    conn = None
    try:
        # 2. CONNECT TO DATABASE
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            connect_timeout=3 # limit timeout to 3 seconds to prevent long hangups
        )
        cur = conn.cursor()

        # 3. EXECUTE SQL QUERY
        query = """
            INSERT INTO upload_logs 
            (timestamp, ip_address, user_device, file_name, file_ext, file_size_mb, target_format, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        data = (timestamp, ip_address, user_agent, filename, file_ext, size_mb, target_format, status)
        cur.execute(query, data)
        
        # 4. SAVE CHANGES
        conn.commit()
        cur.close()

    except Exception as e:
        print(f"!!! DATABASE ERROR: {e}. Falling back to CSV logging.")
        # Fall back to CSV on connection or insertion error
        log_to_csv(timestamp, ip_address, user_agent, filename, file_ext, size_mb, target_format, status)
        
    finally:
        # 5. ALWAYS CLOSE CONNECTION
        if conn is not None:
            conn.close()
