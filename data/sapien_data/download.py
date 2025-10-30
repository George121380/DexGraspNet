import sapien
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InBlaXFpNjIxQGJlcmtlbGV5LmVkdSIsImlwIjoiMTcyLjIwLjAuMSIsInByaXZpbGVnZSI6MSwiZmlsZU9ubHkiOnRydWUsImlhdCI6MTc1OTkyOTk3MywiZXhwIjoxNzkxNDY1OTczfQ.vWAY5J1HZ8bxqEsqwvkMiQXxBMhBXKjUkhxlzqRoqrk"

list = [100015,100017,100021,100023,100025,100028,100032,100038,100040,100045,100047,100051,100054,100057,100058,100060,100623,102080]
for id in list:
    urdf_file = sapien.asset.download_partnet_mobility(id, token)
    print(f"Downloaded {id}")
