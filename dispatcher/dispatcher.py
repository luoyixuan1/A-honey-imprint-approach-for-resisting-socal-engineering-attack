import pymysql
import socket
import json

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='12345678', db='db')
cursor = conn.cursor()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 8888))
s.listen(5)
print("Waiting connection from client...")
conn, address = s.accept()
print("Connected by {}".format(address))
while True:
    receive_message = conn.recv(1024).decode()
    receive_message = json.loads(receive_message)
    # receive_message = {'type': "query message", "content": "id=1"}
    # receive_message = {'type': "SE alert", "content": "suspicious inbound message"}
    results = ''

    if receive_message["type"] == "query message":
            cursor.execute("select * from realInformation where {}".format(receive_message["content"]))
            results = cursor.fetchall()
    elif receive_message["type"] == "SE alert":
        cursor.close()
        conn.close()

        conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='12345678', db='db')
        cursor = conn.cursor()
        cursor.execute("select * from honeyImprintedData where {}".format(receive_message["content"]))
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='12345678', db='db')
        cursor = conn.cursor()

    send_message = json.dumps(results).encode()
    conn.send(send_message)
