from elasticsearch import Elasticsearch

# Connect to the Elasticsearch instance
es = Elasticsearch(["http://localhost:50004"])

# Fetch all index names
all_indices = es.indices.get_alias(name="*").keys()

print(all_indices)

# Delete each index
for index in all_indices:
    es.indices.delete(index=index)
