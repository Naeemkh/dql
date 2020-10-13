import os
import io
import time
import random
import sqlite3
import numpy as np


class DataBase:
    """
    Wraper class for SQlite3 database.
    """

    def __init__(self):
        pass
       
    def connect_to_db(self, db_name=None):
        
        if db_name:        
            self.db_path = db_name
        else: 
            self.db_path = os.path.join(self.db_folder,self.db_name)
           
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.c = self.conn.cursor()
    
    def initialize(self, db_folder, db_name):
        self.db_folder = db_folder
        self.db_name = db_name
        self.connect_to_db()
        self.c.execute('CREATE TABLE IF NOT EXISTS cnndata(uid TEXT, image_arr array, action REAL)')
    
    def close_db_connection(self):
        self.c.close()
        self.conn.close()
        
    def enter_value_db(self, uid, img_arr, action):
        self.c.execute("INSERT INTO cnndata (uid, image_arr, action) VALUES (?, ?, ?)", (uid, img_arr, action))
        self.conn.commit()

    def read_from_db(self, n_data):
        self.c.execute('SELECT * FROM cnndata WHERE rowid IN (SELECT rowid FROM cnndata ORDER BY RANDOM() LIMIT ? )', (n_data,))
        data = self.c.fetchall()
        return data
   
    @staticmethod
    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)


if __name__ == "__main__":
    db = DataBase()
    db.initialize('myfolder8','mydb1.db')
    # db.connect_to_db()
    
    for i in range(100):
        a = random.choice([0,1,2,3])
        b = np.array(np.arange(100))
        c = "QTEYRD162"
        db.enter_value_db(c, b, a)
    
    data = db.read_from_db(5)

    for i in data:
        print(i)

    db.close_db_connection()
