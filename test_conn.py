import os
import pymysql
from dotenv import load_dotenv

load_dotenv()  # legge .env nella cartella corrente

host = os.getenv("DB_HOST")
port = int(os.getenv("DB_PORT", "3306"))
user = os.getenv("DB_USER")
pwd = os.getenv("DB_PASSWORD")
db = os.getenv("DB_NAME")
cs = os.getenv("DB_CHARSET", "utf8mb4")

print("Trying:", host, port, user, db)
try:
    conn = pymysql.connect(host=host, port=port, user=user,
                           password=pwd, db=db, charset=cs)
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        print("OK:", cur.fetchone())
    conn.close()
except Exception as e:
    print("ERROR:", repr(e))
