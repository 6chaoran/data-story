# Set Up Airflow 1.9.0 on ubuntu 16.04 LTS

## Installation
First of all, make sure python2.7 and pip are installed and upgraded to the latest.  
create a directory for Airflow: `mkdir ~/airflow`  
set it as the home directory: `export AIRFLOW_HOME='~/airflow'`  
install using `pip install apache-airflow==1.9.0`, run `sudo pip install apache-airflow==1.9.0` instead, if you encountered permission denied.

once airflow is installed, `airflow version` to confirm the installation.

```
[2018-07-20 06:42:41,919] {driver.py:120} INFO - Generating grammar tables from /usr/lib/python2.7/lib2to3/Grammar.txt
[2018-07-20 06:42:41,949] {driver.py:120} INFO - Generating grammar tables from /usr/lib/python2.7/lib2to3/PatternGrammar.txt
[2018-07-20 06:42:42,247] {configuration.py:206} WARNING - section/key [celery/celery_ssl_active] not found in config
[2018-07-20 06:42:42,247] {default_celery.py:41} WARNING - Celery Executor will run without SSL
[2018-07-20 06:42:42,248] {__init__.py:45} INFO - Using executor CeleryExecutor
  ____________       _____________
 ____    |__( )_________  __/__  /________      __
____  /| |_  /__  ___/_  /_ __  /_  __ \_ | /| / /
___  ___ |  / _  /   _  __/ _  / / /_/ /_ |/ |/ /
 _/_/  |_/_/  /_/    /_/    /_/  \____/____/|__/
   v1.9.0
```

run `airflow initdb` will create a `airflow.db` file
```
[2018-07-20 06:56:27,015] {driver.py:120} INFO - Generating grammar tables from /usr/lib/python2.7/lib2to3/Grammar.txt
[2018-07-20 06:56:27,037] {driver.py:120} INFO - Generating grammar tables from /usr/lib/python2.7/lib2to3/PatternGrammar.txt
[2018-07-20 06:56:27,322] {configuration.py:206} WARNING - section/key [celery/celery_ssl_active] not found in config
[2018-07-20 06:56:27,322] {default_celery.py:41} WARNING - Celery Executor will run without SSL
[2018-07-20 06:56:27,323] {__init__.py:45} INFO - Using executor CeleryExecutor
DB: postgresql+psycopg2://ubuntu@localhost:5432/airflow
[2018-07-20 06:56:27,502] {db.py:312} INFO - Creating tables
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
[2018-07-20 06:56:27,709] {models.py:189} INFO - Filling up the DagBag from /home/lac-user/airflow/ntuc_core_pipelines
[2018-07-20 06:56:27,720] {models.py:2191} WARNING - schedule_interval is used for <Task(BashOperator): print_result>, though it has been deprecated as a task parameter, you need to specify it as a DAG parameter instead
Done.
```

run `airflow scheduler` to start airflow scheduler. 

```
[2018-07-20 07:34:00,013] {driver.py:120} INFO - Generating grammar tables from /usr/lib/python2.7/lib2to3/Grammar.txt
[2018-07-20 07:34:00,035] {driver.py:120} INFO - Generating grammar tables from /usr/lib/python2.7/lib2to3/PatternGrammar.txt
[2018-07-20 07:34:00,312] {configuration.py:206} WARNING - section/key [celery/celery_ssl_active] not found in config
[2018-07-20 07:34:00,313] {default_celery.py:41} WARNING - Celery Executor will run without SSL
[2018-07-20 07:34:00,313] {__init__.py:45} INFO - Using executor CeleryExecutor
  ____________       _____________
 ____    |__( )_________  __/__  /________      __
____  /| |_  /__  ___/_  /_ __  /_  __ \_ | /| / /
___  ___ |  / _  /   _  __/ _  / / /_/ /_ |/ |/ /
 _/_/  |_/_/  /_/    /_/    /_/  \____/____/|__/

[2018-07-20 07:34:00,403] {jobs.py:1507} INFO - Starting the scheduler
[2018-07-20 07:34:00,404] {jobs.py:1520} INFO - Processing files using up to 2 processes at a time
[2018-07-20 07:34:00,404] {jobs.py:1521} INFO - Running execute loop for -1 seconds
[2018-07-20 07:34:00,404] {jobs.py:1522} INFO - Processing each file at most -1 times
[2018-07-20 07:34:00,404] {jobs.py:1523} INFO - Process each file at most once every 0 seconds
[2018-07-20 07:34:00,404] {jobs.py:1524} INFO - Checking for new files in /home/lac-user/airflow/ntuc_core_pipelines every 300 seconds
[2018-07-20 07:34:00,404] {jobs.py:1527} INFO - Searching for files in /home/lac-user/airflow/ntuc_core_pipelines
[2018-07-20 07:34:00,408] {jobs.py:1529} INFO - There are 7 files in /home/lac-user/airflow/ntuc_core_pipelines
[2018-07-20 07:34:00,408] {jobs.py:1588} INFO - Resetting orphaned tasks for active dag runs
[2018-07-20 07:34:00,414] {jobs.py:1627} INFO - Heartbeating the process manager

```

run `airflow webserver -p 8080` to start airflow webUI on port :8080, you will be able to open webUI at `http://localhost:8080`. You will need set your SSH port forwarding, if  setup is done on remote server.(e.g. Windows: Putty > Connection > SSH > Auth > Tunnels > Add new forwarded port, Mac: ssh user@server.ip -L 8080:localhost:8080)


### Tips
1. run airflow process with `-D` flag so that the process will be daemonize, which means will run in background.
for example:
```
airflow scheduler -D
airflow webserver -p 8080 -D
```

2. airflow webserver will have multiple (4 by default) workers, killing webserver workers process if you want to restart/shutdown your airflow. for example:
```
ps -ef | grep airflow
sudo kill -9 airflow-pids
```

3. can't daemonize the process

try delete the `.err` and `.pid` file in airflow folder. for example:
```
cd ~/airflow
rm airflow-webserver.err airflow-webserver.pid
```

4. DAGs triggered but never run or in queue

 * make sure airflow scheduler is running  
 * make sure airflow workers are running
 * make sure the pause toggle on the left side is turned on


## Executor Mode
Airflow offer different execution mode for different scenarios:

1. Sequential Executor

	single machine, single-process. Airflow out-of-box option.
    
2. Local Executor

	single machine, multi-process.
	need to use `postgres` database instead of sqlite
    
3. Celery Executor

	could be distributed in multi-machine. need to start workers to pick up tasks.

    
### Postgres Database Setup
install postgres: `sudo apt-get install postgresql postgresql-contrib`  
sign in database with superuser `postgres`: `sudo -u postgres psql`

```
CREATE ROLE ubuntu;
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES on database airflow to ubuntu;
ALTER ROLE ubuntu SUPERUSER;
ALTER ROLE ubuntu CREATEDB;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ubuntu;
```
remember to end the syntax with `;`   
use `\du` to confirm the role `ubuntu` is properly set:
```
postgres=# \du
                                   List of roles
 Role name |                         Attributes                         | Member of
-----------+------------------------------------------------------------+-----------
 postgres  | Superuser, Create role, Create DB, Replication, Bypass RLS | {}
 ubuntu    | Superuser, Create DB                                       | {}
```

change the listen port: `/etc/postgresql/9.*/main/pg_hba.conf`
	
* ipv4 address to `0.0.0.0/0`
* ipv4 connection `md5` to `trust`

start the postgres db: `sudo service postgresql start`

## Reference
[Installing Apache Airflow on Ubuntu/AWS – A.R.G.O. – Medium](https://medium.com/a-r-g-o/installing-apache-airflow-on-ubuntu-aws-6ebac15db211)
