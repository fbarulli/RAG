---
id: 6a3a0cc252
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_c66aba56.png
question: Login to DB not working in Adminer UI even after right DB, user and password.
sort_order: 28
---

**Problem:** Adminer UI is not responding or showing database details, even with the correct database, user, and password.

<{IMAGE:image_1}>

---

**Solution:** Try accessing the database from the command line using `psql`.

You can quickly install `psql` via package managers such as `sudo apt`.

Here is an example:

```bash
(base) cpl@inpne-ed-lab003:~$ psql -h localhost -p 5432 -U postgres

Password for user postgres: 

psql (14.12 (Ubuntu 14.12-0ubuntu0.22.04.1), server 16.4 (Debian 16.4-1.pgdg120+1))

WARNING: psql major version 14, server major version 16.
Some psql features might not work.

Type "help" for help.

postgres=# \l

List of databases

 Name    |  Owner   | Encoding |  Collate   |   Ctype    |   Access privileges 
-----------+----------+----------+------------+------------+-----------------------
postgres  | postgres | UTF8     | en_US.utf8 | en_US.utf8 | 
template0 | postgres | UTF8     | en_US.utf8 | en_US.utf8 | =c/postgres          +
          |          |          |            |            | postgres=CTc/postgres
template1 | postgres | UTF8     | en_US.utf8 | en_US.utf8 | =c/postgres          +
          |          |          |            |            | postgres=CTc/postgres
test      | postgres | UTF8     | en_US.utf8 | en_US.utf8 |
(4 rows)
```