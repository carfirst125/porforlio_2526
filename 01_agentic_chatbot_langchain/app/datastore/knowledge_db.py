
import os
import json
import psycopg2
from psycopg2 import sql
import pandas as pd
import numpy as np
import datetime
import uuid
import traceback  # để in stack trace chi tiết
from app.utils.clients import DBConnection
import logging


logging.basicConfig(
    level=logging.INFO,  # hoặc DEBUG nếu bạn cần chi tiết hơn
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # In ra terminal
)
logger = logging.getLogger(__name__)


class GetKnowledgeDB:
    """
    A utility class for connecting to and querying a PostgreSQL knowledge database.

    This class handles establishing a connection to the PostgreSQL database
    using the credentials stored in `DBConnection.agentic_db_dict`, 
    and provides a method for executing SQL queries and returning results as a pandas DataFrame.

    Attributes:
        conn (psycopg2.extensions.connection): Active database connection object.
        cursor (psycopg2.extensions.cursor): Cursor object for executing SQL queries.

    Methods:
        _db_connect():
            Establishes a connection to the PostgreSQL database using the provided configuration.
        query_db(sql: str) -> pandas.DataFrame:
            Executes the given SQL query and returns the result as a pandas DataFrame.
    """
    def __init__(self):
        self.conn = ''
        self.cursor = ''
        self._db_connect()
        
    def _db_connect(self):
        
        # Thông tin kết nối
        db_config = {
            'host'    : DBConnection.agentic_db_dict["DBHOST"],
            'port'    : DBConnection.agentic_db_dict["DBPORT"],
            'dbname'  : DBConnection.agentic_db_dict["DBNAME"],
            'user'    : DBConnection.agentic_db_dict["DBUSER"],
            'password': DBConnection.agentic_db_dict["DBPASS"]
        }
        
        # Kết nối đến PostgreSQL
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        
    
    def query_db(self, sql):

        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        df = pd.DataFrame(data=result)    
        df.columns = [x[0] for x in self.cursor.description]
        
        return df
    
        
class GetDBSchemaTable:
    """Get DB Schema and Table list which will be migrated based on sweeping on Source DB
        Attributes:
        DB_MIGRATE_TABLES  : migrate table sweeping from Source DB, data from those table will be migrate to new DB
        EXTEND_EMPTY_TABLES: new tables create in new Python 
    """
    
    def __init__(self):
        self.DB_MIGRATE_TABLES = self.db_get_schema()
        self.DB_MIGRATE_TABLES['public'] = ['oai_embedding','oai_prompt','oai_abbr'] # con oai_settings need process because changed table.schema
        self.EXTEND_EMPTY_TABLES = {'public': ['oai_setting','log','log_emb','chats','sources','messages']}
    
    def db_get_schema(self):
        """sweep db get schema and table names in schema
           Args:
            db_key_dict (dict): connection authorization
           Return:
            db_schema_tb (dict): schema hierarchy {"schema1":["table_name1","table_name2","table_name3"],"schema2":[]}
        """
        
        # Thông tin kết nối
        db_config = {
            'host'    : DBConnection.src_db_dict["DBHOST"],
            'port'    : DBConnection.src_db_dict["DBPORT"],
            'dbname'  : DBConnection.src_db_dict["DBNAME"],
            'user'    : DBConnection.src_db_dict["DBUSER"],
            'password': DBConnection.src_db_dict["DBPASS"]
        }
        
        db_schema_tb = {}
        
        try:
            # Kết nối đến PostgreSQL
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Truy vấn danh sách schema (bỏ các schema hệ thống như pg_catalog, information_schema)
            cursor.execute("""
                SELECT schema_name 
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schema_name;
            """)

            # In danh sách schema
            schemas = cursor.fetchall()
            for schema in schemas:
                print(schema[0])
                table_list = []

                schema_name = schema[0]
                print(f"\nSchema: {schema_name}")

                # 2. Lấy danh sách các bảng trong schema
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                """, (schema_name,))

                tables = cursor.fetchall()
                if not tables:
                    print("  (No tables)")
                else:
                    for table in tables:
                        print(f"  - {table[0]}")
                        table_list.append(table[0])
                    
                db_schema_tb[schema[0]] = table_list

        except Exception as e:
            logger.error(f"[db_get_schema][Exception] Error: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                
        return db_schema_tb
        

class CreateMigrateTable(GetDBSchemaTable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.migrate_schema = self.DB_MIGRATE_TABLES
        self.extent_schema = self.EXTEND_EMPTY_TABLES
        self.create_table_tuple = ('oai_embedding','oai_prompt','oai_setting','oai_abbr','fm_ccy_rate','tbl_saving_core_rate','log','log_emb','chats','messages','sources')
    
    #-- Bảng oai_embedding
    create_oai_embedding = """
    CREATE TABLE IF NOT EXISTS public.oai_embedding (
        id UUID PRIMARY KEY NOT NULL,
        source VARCHAR(500) NOT NULL,
        input VARCHAR(20000) NOT NULL,
        tokens INTEGER NOT NULL,
        status VARCHAR(1) NOT NULL,
        create_at TIMESTAMPTZ NOT NULL,
        context_uid VARCHAR(20) NOT NULL,
        web_url VARCHAR(500) NOT NULL,
        type CHAR(1) NOT NULL,
        embedding VECTOR NOT NULL
    );
    """
    
    #-- Bảng oai_prompt
    create_oai_prompt = """
    CREATE TABLE IF NOT EXISTS public.oai_prompt (
        id UUID PRIMARY KEY NOT NULL,
        question VARCHAR(1000) NOT NULL,
        answer VARCHAR(5000) NOT NULL,
        create_at TIMESTAMPTZ NOT NULL,
        status VARCHAR(1) NOT NULL,
        embedding VECTOR NOT NULL,
        create_by VARCHAR(20),
        type VARCHAR(20)
    );
    """
    
    #-- Bảng oai_abbr
    create_oai_abbr = """
    CREATE TABLE IF NOT EXISTS public.oai_abbr (
        id UUID PRIMARY KEY NOT NULL,
        full_word VARCHAR(100) NOT NULL,
        create_at TIMESTAMPTZ NOT NULL,
        type VARCHAR(1) NOT NULL,
        status VARCHAR(1) NOT NULL,
        regex JSON NOT NULL,
        create_by VARCHAR(20)
    );
    """
    
    #-- Bảng fm_ccy_rate
    create_fm_ccy_rate = """
    CREATE TABLE IF NOT EXISTS eoc.fm_ccy_rate (
        ccy VARCHAR(3) NOT NULL,
        branch VARCHAR(6) NOT NULL,
        rate_type VARCHAR(3) NOT NULL,
        effective_date TIMESTAMPTZ NOT NULL,
        last_change_date TIMESTAMPTZ NOT NULL,
        quote_type VARCHAR(1) NOT NULL,
        ccy_rate DOUBLE PRECISION NOT NULL,
        buy_rate DOUBLE PRECISION NOT NULL,
        sell_rate DOUBLE PRECISION NOT NULL,
        central_bank_rate DOUBLE PRECISION,
        buy_spread DOUBLE PRECISION,
        sell_spread DOUBLE PRECISION,
        ctrl_spread_usd DOUBLE PRECISION,
        cdc_id BIGINT NOT NULL,
        cdc_timestamp TIMESTAMPTZ,
        PRIMARY KEY (ccy, branch, rate_type)
    );
    """
    
    #-- Bảng tbl_saving_core_rate
    create_tbl_saving_core_rate = """
    CREATE TABLE IF NOT EXISTS eocetl.tbl_saving_core_rate (
        id UUID PRIMARY KEY NOT NULL,
        branch VARCHAR(6),
        ccy VARCHAR(3),
        term VARCHAR(5),
        day_basis SMALLINT,
        balance BIGINT,
        int_type VARCHAR(3),
        cr_int_freq VARCHAR(5),
        actual_rate DOUBLE PRECISION,
        effect_date TIMESTAMP(6) WITHOUT TIME ZONE,
        last_change_date TIMESTAMP(6) WITHOUT TIME ZONE,
        last_update TIMESTAMP(6) WITHOUT TIME ZONE,
        product_type VARCHAR(3),
        int_basis VARCHAR(3)
    );
    """

    #-- Bảng OAI_SETTING
    create_oai_setting = """
    CREATE TABLE IF NOT EXISTS public.oai_setting (
        id VARCHAR(255) PRIMARY KEY NOT NULL,
        create_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        update_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        config JSON
    );
    """

    #-- Bảng LOG
    create_log = """
    CREATE TABLE IF NOT EXISTS public.log (
        id UUID PRIMARY KEY NOT NULL,
        user_id VARCHAR(50),
        chat_id UUID NOT NULL,
        message_id UUID NOT NULL,
        raw_input VARCHAR(500) NOT NULL,
        normalized_input VARCHAR(500),
        channel VARCHAR(20),
        vs_time_ms DOUBLE PRECISION,
        vs_tokens INTEGER,
        intent_time_ms DOUBLE PRECISION,
        intent_tokens INTEGER,
        intent_result VARCHAR(50),
        intent_confidence DOUBLE PRECISION,
        intent_finish_reason VARCHAR(50),
        function_name VARCHAR(100),
        function_params JSON,
        function_time_ms DOUBLE PRECISION,
        function_tokens INTEGER,
        standalone_time_ms DOUBLE PRECISION,
        standalone_tokens INTEGER,
        standalone_result VARCHAR(1000),
        standalone_finish_reason VARCHAR(50),
        qa_distance DOUBLE PRECISION,
        doc_count INTEGER,
        answer_time_ms DOUBLE PRECISION,
        answer_tokens INTEGER,
        answer_result VARCHAR(1000),
        answer_finish_reason VARCHAR(50),
        embedding_model VARCHAR(100),
        classification_completion_model VARCHAR(100),
        standalone_completion_model VARCHAR(100),
        qa_completion_model VARCHAR(100),
        error VARCHAR(500),
        created_at TIMESTAMPTZ NOT NULL,
        additional_metadata JSON
    );
    """

    # -- Bảng LOG_EMB
    create_log_emb = """
    CREATE TABLE IF NOT EXISTS public.log_emb (
        id UUID PRIMARY KEY NOT NULL,
        log_id UUID REFERENCES public.log(id),
        embedding_id UUID NOT NULL,
        create_at TIMESTAMPTZ NOT NULL,
        distance DOUBLE PRECISION
    );
    """

    #-- Bảng MESSAGES
    create_messages = """
    CREATE TABLE IF NOT EXISTS public.messages (
        id UUID PRIMARY KEY NOT NULL,
        user_id VARCHAR,
        chat_id UUID REFERENCES public.chats(id),
        human_content VARCHAR NOT NULL,
        ai_content VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    );
    """


    #-- Bảng CHATS
    create_chats = """
    CREATE TABLE IF NOT EXISTS public.chats (
        id UUID PRIMARY KEY NOT NULL,
        user_id VARCHAR,
        chat_title VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL,
        updated_at VARCHAR NOT NULL
    );
    """

    #-- Bảng SOURCES
    create_sources = """
    CREATE TABLE IF NOT EXISTS public.sources (
        id UUID PRIMARY KEY NOT NULL,
        message_id UUID REFERENCES public.messages(id),
        page_content VARCHAR NOT NULL,
        cmetadata JSONB,
        created_at VARCHAR NOT NULL
    );
    """
    
    def create_schema(self):
    
        # Kết nối đến PostgreSQL
        target_conn = psycopg2.connect(
             host  =     DBConnection.agentic_db_dict["DBHOST"],
             port  =     DBConnection.agentic_db_dict["DBPORT"],
             dbname=     DBConnection.agentic_db_dict["DBNAME"],
             user  =     DBConnection.agentic_db_dict["DBUSER"],
             password=   DBConnection.agentic_db_dict["DBPASS"]
        )
        target_cursor = target_conn.cursor()
            
        for schema in self.migrate_schema.keys():
            logger.info(f"[CreateMigrateTable.create_schema] Creating schema: {schema}")
            
            # Bước 2: Tạo schema trên DB đích nếu chưa có
            target_cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
            target_conn.commit()
    
    
    def create_table(self, create_sql_str):
        
        try:
            # Kết nối đến PostgreSQL
            conn = psycopg2.connect(
                host  =     DBConnection.agentic_db_dict["DBHOST"],
                port  =     DBConnection.agentic_db_dict["DBPORT"],
                dbname=     DBConnection.agentic_db_dict["DBNAME"],
                user  =     DBConnection.agentic_db_dict["DBUSER"],
                password=   DBConnection.agentic_db_dict["DBPASS"]
            )
            cursor = conn.cursor()
            
            ## 1. Tạo extension vector nếu chưa có
            #cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            #conn.commit()
            
            cursor.execute(create_sql_str)        
            conn.commit()

            # Đóng kết nối
            cursor.close()
            conn.close()       
            
        except Exception as e:
            logger.error("[create_table][Exception] Try to connect db:", str(e))
            traceback.print_exc()  # In stack trace để debug kỹ hơn
            return False
        
        return True

    def create_db_tables(self):
        # create scheme if not exist
        self.create_schema()
        # create tables if not exist
        for tb in self.create_table_tuple:   
            logger.info(f"[create_tb_tables] create {tb}...")
            #value = globals()[f"create_{tb}"]
            create_sql = getattr(self.__class__, f"create_{tb}")
            logger.info(f"[create_tb_tables] create \n {create_sql}...")
            response = self.create_table(create_sql)
            if response:
                logger.info(f"[create_tb_tables] create {tb} Completed")

    
    
class TableMigrate(GetDBSchemaTable):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.migrate_schema = self.DB_MIGRATE_TABLES
        self.extent_schema = self.EXTEND_EMPTY_TABLES

        # Kết nối DB nguồn (A)
        self.source_conn = psycopg2.connect(
            host  =     DBConnection.src_db_dict["DBHOST"],
            port  =     DBConnection.src_db_dict["DBPORT"],
            dbname=     DBConnection.src_db_dict["DBNAME"],
            user  =     DBConnection.src_db_dict["DBUSER"],
            password=   DBConnection.src_db_dict["DBPASS"]
        )
        
                # Kết nối DB đích (B)
        self.target_conn = psycopg2.connect(
            host  =     DBConnection.agentic_db_dict["DBHOST"],
            port  =     DBConnection.agentic_db_dict["DBPORT"],
            dbname=     DBConnection.agentic_db_dict["DBNAME"],
            user  =     DBConnection.agentic_db_dict["DBUSER"],
            password=   DBConnection.agentic_db_dict["DBPASS"]
        )
        
        self.source_cursor = self.source_conn.cursor()
        self.target_cursor = self.target_conn.cursor()
        
    def db_migration(self):
        """migrate all tables defined in the migrate schema from source to destination
        
        Args:
            src_db_key_dict
            dst_db_key_dict
            migrate_schema
            
        Output:
            data migration on destination db
        """
        
        # # Bước 0: Tạo extension dblink nếu chưa có
        # try:
        #     target_cursor.execute("CREATE EXTENSION IF NOT EXISTS dblink;")
        #     target_conn.commit()
        # except Exception as e:
        #     print("❌ Lỗi khi tạo extension dblink:", e)
        #     target_conn.rollback()
        #     sys.exit(1)

        # Bước 1: Lấy danh sách schema (ngoại trừ schema hệ thống)
        self.source_cursor.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
        """)
        schemas = [row[0] for row in self.source_cursor.fetchall()]
        # schemas = ['public']
        
        for schema in schemas:
            print(f"Đang xử lý schema: {schema}")
            if schema in self.migrate_schema.keys():
                
                # Bước 2: Tạo schema trên DB đích nếu chưa có
                self.target_cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
                self.target_conn.commit()
                
                # Bước 3: Lấy danh sách các bảng trong schema
                self.source_cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_type = 'BASE TABLE'
                """, (schema,))
                tables = [row[0] for row in self.source_cursor.fetchall()]
                print(f"tables: {tables}")
                
                for table in tables:
                    print(f"  -> Di chuyển bảng: {schema}.{table}")
                    if table in self.migrate_schema[schema]:
                        self.target_cursor.execute(f"""SELECT count(*) FROM {schema}.{table}""")
                        if self.target_cursor.fetchall()[0][0]:
                            continue
                        
                        print(f"  ---> Processing ...")
                        # # Bước 4: Tạo bảng mới và chèn dữ liệu từ dblink (nếu cần dùng dblink)
                        # create_table_sql = sql.SQL("""
                        #     CREATE TABLE IF NOT EXISTS {schema}.{table} AS
                        #     SELECT * FROM dblink('dbname={src_db} host={src_host} user={src_user} password={src_pass}',
                        #                         'SELECT * FROM {schema}.{table}')
                        #     AS t1(*)
                        # """).format(
                        #     schema=sql.Identifier(schema),
                        #     table=sql.Identifier(table),
                        #     src_db=sql.Literal(src_key_dict["DBNAME"]),
                        #     src_host=sql.Literal(src_key_dict["DBHOST"]),
                        #     src_user=sql.Literal(src_key_dict["DBUSER"]),
                        #     src_pass=sql.Literal(src_key_dict["DBPASS"])
                        # )

                        # # Có thể thực thi create_table_sql nếu muốn tạo bảng từ dblink
                        # target_cursor.execute(create_table_sql)                                                

                        # Lấy dữ liệu
                        self.source_cursor.execute(sql.SQL("SELECT * FROM {}.{}").format(
                            sql.Identifier(schema), sql.Identifier(table)))
                        rows = self.source_cursor.fetchall()
                        columns = [desc[0] for desc in self.source_cursor.description]

                        # Xác định index của các cột cần convert về JSON nếu cần
                        json_columns = {'regex'}  # danh sách các cột cần ép kiểu json
                        json_column_indices = [i for i, col in enumerate(columns) if col in json_columns]

                        # Tạo câu lệnh INSERT
                        insert_query = sql.SQL("""
                            INSERT INTO {}.{} ({})
                            VALUES ({})
                        """).format(
                            sql.Identifier(schema),
                            sql.Identifier(table),
                            sql.SQL(', ').join(map(sql.Identifier, columns)),
                            sql.SQL(', ').join(sql.Placeholder() * len(columns))
                        )

                        # Thực hiện insert từng dòng
                        for row in rows:
                            row = list(row)  # chuyển từ tuple sang list để gán lại giá trị
                            for idx in json_column_indices:
                                if row[idx] is not None:
                                    row[idx] = json.dumps(row[idx])  # chuyển sang chuỗi JSON
                            self.target_cursor.execute(insert_query, row)

                        self.target_conn.commit()                                                                 
                        
        # Đóng kết nối
        # self.source_cursor.close()
        # self.target_cursor.close()
        # self.source_conn.close()
        # self.target_conn.close()

        logger.info("[TableMigrate.db_migration] Data Migration completed!")
        
    ################################
    def db_migration_cleanup(self):
        
        for schema in self.migrate_schema.keys():
            logger.info(f"[db_migration_cleanup] cleaning schema = {schema}")
            if schema == 'public':
                for table in self.migrate_schema[schema]:
                    logger.info(f"[db_migration_cleanup] cleaning {schema}.{table}")
                    self.target_cursor.execute(f"DROP TABLE IF EXISTS {schema}.{table} CASCADE;")
                    self.target_conn.commit()
                    
            else:
                logger.info(f"[db_migration_cleanup] cleaning {schema}")
                self.target_cursor.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
                self.target_conn.commit()
                
        for schema in self.extend_schema.keys():
            for table in self.extend_schema[schema][::-1]:
                logger.info(f"[db_migration_cleanup] cleaning {schema}.{table}")
                self.target_cursor.execute(f"DROP TABLE IF EXISTS {schema}.{table} CASCADE;")
                self.target_conn.commit()
    
    ##################################
    def insert_oai_setting(self, id_val='config'):
        """insert data for oai_setting
           The data of oai_setting is modified in comparison to origin.
           The column "config having value stored in ../config/configuration.json"
           This function is particular process for data of oai_setting
        
        """
        
        # --- Dữ liệu cần insert ---
        now = datetime.datetime.now()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, '..', 'config', 'configuration.json')
        with open(config_path, 'r', encoding="utf-8") as file:
            oai_setting_config_json = json.loads(file.read())

        # --- Tạo bảng nếu chưa có ---
        create_table_query = """
        CREATE TABLE IF NOT EXISTS public.oai_setting (
            id VARCHAR(255) PRIMARY KEY NOT NULL,
            create_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            update_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            config JSON
        );
        """
        
        print("✅ Create 'oai_setting' if have not existed yet.")
        self.target_cursor.execute(create_table_query)
        self.target_conn.commit()

        # --- Câu lệnh SQL ---
        insert_query = """
        INSERT INTO public.oai_setting (id, create_at, update_at, config)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            update_at = EXCLUDED.update_at,
            config = EXCLUDED.config;
        """

        self.target_cursor.execute(
            insert_query,
            (
                id_val,                  # id
                now,                     # create_at
                now,                     # update_at
                json.dumps(oai_setting_config_json, ensure_ascii=False, indent=6)  # config (dưới dạng string JSON)
            )            
        )

        self.target_conn.commit()
        print("✅ insert data to TABLE oai_setting completed.")

    ##################################
    def update_qtype_examples(self):
        """update oai_prompt table 
        add new question type to file question_type_examples.xlsx and rerun python -m app.main with  TableMigrate_obj.update_qtype_examples()

        """
        self.target_cursor.execute("select count(*) from oai_prompt")
        self.target_conn.commit()
        
        result = self.target_cursor.fetchall()    
        df_count = pd.DataFrame(data=result)
        df_count.columns = [x[0] for x in self.target_cursor.description]
        print(df_count.loc[0,'count'])
        qtype_count = df_count.loc[0,'count']
        df = pd.read_excel("app/config/question_type_examples.xlsx", index_col=False)
        if len(df) > qtype_count:
            df_process = df[qtype_count:]
            df_process = df_process.fillna("")
            
        for i, row in df_process.iterrows():
            print(i,'--', row.to_dict())
            self.insert_oai_prompt(row.to_dict())          
            
    ##################################
    def insert_oai_prompt(self, data_row):
        """insert data for oai_prompt
        insert one qtype example row for every call
        Args: 
            data_row (dict): data row dictionary
        """

        print(f"data_row:{data_row}")
        embedding_vt = BedrockLLM.get_embedding_vector(data_row['question'])        
        print(f"embedding_vt : {embedding_vt}")
        # Tạo các giá trị khác
        record_id = str(uuid.uuid4())
        create_at = datetime.datetime.now(datetime.timezone.utc)
        answer = data_row['expected_answer']
        status = "A"  # ví dụ trạng thái A = Active
        create_by = None  # hoặc chuỗi user nào đó

        # Câu lệnh INSERT
        sql = """
        INSERT INTO oai_prompt (id, question, answer, create_at, status, embedding, create_by, type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Thực thi
        self.target_cursor.execute(sql, (
            record_id,
            data_row["question"],
            answer,
            create_at,
            status,
            embedding_vt,  # pgvector chấp nhận list[float]
            create_by,
            data_row["question_type"]
        ))

        self.target_conn.commit()

        print("✅ Insert thành công")
