import sapien
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InBlaXFpNjIxQGJlcmtlbGV5LmVkdSIsImlwIjoiMTcyLjIwLjAuMSIsInByaXZpbGVnZSI6MSwiZmlsZU9ubHkiOnRydWUsImlhdCI6MTc1OTkyOTk3MywiZXhwIjoxNzkxNDY1OTczfQ.vWAY5J1HZ8bxqEsqwvkMiQXxBMhBXKjUkhxlzqRoqrk"

list = [
    3386,
    3393,
    4094,
    4529,
    4533,
    4541,
    4542,
    4552,
    4562,
    4563,
    4564,
    4566,
    4571,
    4574,
    4576,
    4578,
    4586,
    4589,
    4590,
    4592,
    4594,
    4627,
    4628,
    4633,
    4681,
    4853,
    5050,
    5306,
    5477
]
for id in list:
    urdf_file = sapien.asset.download_partnet_mobility(id, token)
    print(f"Downloaded {id}")
