import pymongo

uri = "mongodb://leonardoserrate9_db_user:tw7cUksqJnILljES@ac-rjpknn9-shard-00-00.8n1986b.mongodb.net:27017,ac-rjpknn9-shard-00-01.8n1986b.mongodb.net:27017,ac-rjpknn9-shard-00-02.8n1986b.mongodb.net:27017/?tls=true&authSource=admin&replicaSet=atlas-r5g2lc-shard-0&retryWrites=true&w=majority&appName=leosb"
client = pymongo.MongoClient(uri)
db = client["politicas_negocio_db"]
user = db.usuarios.find_one({"correo": "leonardoserrate9@gmail.com"})
if user:
    print("USER_ID_FOUND=" + str(user["_id"]))
else:
    print("USER NOT FOUND")
