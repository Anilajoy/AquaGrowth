
import sqlite3

# Establish connection
sql_connection = sqlite3.connect('studyDemo2.db')
cursor = sql_connection.cursor()
print('DBinit')

# Create table query with auto-incrementing id
#create_qry_feedback = '''CREATE TABLE IF NOT EXISTS selectDHprediction (
                   # id INTEGER PRIMARY KEY AUTOINCREMENT,
                  #  DHselection TEXT,
                  #  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                 #   );'''


# Execute query
# cursor.execute(create_qry_feedback)
#print("Table created successfully")
# Create table query with auto-incrementing id
create_qry_diameter = '''CREATE TABLE PlantDiameter2 (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         num_leaves INTEGER,
                         plant_height REAL,
                         ambient_temperature REAL,
                         ambient_humidity REAL,
                         predicted_diameter REAL,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                         );'''




'''CREATE TABLE IF NOT EXISTS selectDHprediction (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    DHselection TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    );'''

# Execute query
cursor.execute(create_qry_diameter)
print("Table created successfully")

# Commit changes
sql_connection.commit()

# Close cursor and connection
cursor.close()
sql_connection.close()