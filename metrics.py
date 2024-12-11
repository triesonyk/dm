import re
import logging
import pandas as pd

from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, Boolean, BigInteger, TIMESTAMP, func, ForeignKey, text

Base = declarative_base()

class AgentInfo(Base):
    __tablename__ = "agent_info"
    __table_args__ = {"schema" : "aidm_webview"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    agent_code = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    total_database = Column(Integer, default=None)
    total_table = Column(Integer, default=None)
    total_data = Column(BigInteger, default=None)
    is_active = Column(Boolean, default=True)
    created_date = Column(TIMESTAMP(timezone=True), server_default=func.now())
    modified_date = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    databases = relationship("DatabaseInfo", back_populates="agent", passive_deletes=True)
    tables = relationship("TableInfo", back_populates="agent", passive_deletes=True)
    
class DatabaseInfo(Base):
    __tablename__ = "database_info"
    __table_args__ = {"schema": "aidm_webview"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)
    hostname = Column(String(255))
    port = Column(Integer)
    instance = Column(String(255))
    location = Column(String(255))
    id_agent = Column(Integer, ForeignKey("aidm_webview.agent_info.id", ondelete="SET NULL"))
    description = Column(Text)
    created_date = Column(TIMESTAMP(timezone=True), server_default=func.now())
    modified_date = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    agent = relationship("AgentInfo", back_populates="databases")
    schemas = relationship("SchemaInfo", back_populates="databases", passive_deletes=True)
    tables = relationship("TableInfo", back_populates="databases", passive_deletes=True)
    
class SchemaInfo(Base):
    __tablename__ = "schema_info"
    __table_args__ = {"schema":"aidm_webview"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    id_database = Column(Integer, ForeignKey("aidm_webview.database_info.id", ondelete="SET NULL"))
    created_date = Column(TIMESTAMP(timezone=True), server_default=func.now())
    modified_date = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    databases = relationship("DatabaseInfo", back_populates="schemas", passive_deletes=True)
    tables = relationship("TableInfo", back_populates="schemas", passive_deletes=True)
    
class TableInfo(Base):
    __tablename__ = "table_info"
    __table_args__ = {"schema":"aidm_webview"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    id_schema = Column(Integer, ForeignKey("aidm_webview.schema_info.id", ondelete="SET NULL"))
    schema_name = Column(String(255), nullable=False)
    id_database = Column(Integer, ForeignKey("aidm_webview.database_info.id", ondelete="SET NULL"))
    database_name = Column(String(255), nullable=False)
    id_agent = Column(Integer, ForeignKey("aidm_webview.agent_info.id", ondelete="SET NULL"))
    agent_name = Column(String(255), nullable=False)
    total_columns = Column(Integer)
    total_rows = Column(BigInteger)
    table_size = Column(BigInteger)
    created_date = Column(TIMESTAMP(timezone=True), server_default=func.now())
    modified_date = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    databases = relationship("DatabaseInfo", back_populates="tables", passive_deletes=True)
    schemas = relationship("SchemaInfo", back_populates="tables", passive_deletes=True)
    agent = relationship("AgentInfo", back_populates="tables", passive_deletes=True)
    
class DataService:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=self.engine)
        
    def add_agent(self, name, agent_code, description=None):
        session = self.Session()
        try:
            session.execute("SET TIMEZONE TO 'Asia/Jakarta';")
            new_agent = AgentInfo(
                name=name,
                agent_code=agent_code,
                description=description
            )
            session.add(new_agent)
            session.commit()
            print(f"Agent '{name}' added successfully with ID {new_agent.id}.")
            return new_agent.id
        except IntegrityError as ie:
            session.rollback()
            print(f"Integrity Error: {ie.orig}")
        except Exception as e:
            session.rollback()
            print(f"An error occurred: {e}")
        finally:
            session.close()
            
    def add_database(self, name, type, id_agent=None, hostname=None, port=None, instance=None, location=None, description=None):
        session = self.Session()
        try:
            session.execute("SET TIMEZONE TO 'Asia/Jakarta';")
            new_database = DatabaseInfo(
                name=name,
                type=type,
                id_agent=id_agent,
                hostname=hostname,
                port=port,
                instance=instance,
                location=location,
                description=description
            )
            session.add(new_database)
            session.commit()
            print(f"Database '{name} added successfully with ID {new_database.id}'")
            return new_database.id
        except IntegrityError as ie:
            session.rollback()
            print(f"Integrity Error: {ie.orig}")
        except Exception as e:
            session.rollback()
            print(f"An error occured: {e}")
        finally:
            session.close()
            
    def add_schema(self, schemas, id_database):
        session = self.Session()
        try:
            for schema in schemas:
                name = schema["schema_name"]
                existing_schema = session.query(SchemaInfo).filter_by(name=name, id_database=id_database).first()
                if existing_schema:
                    print(f"Schema '{name}' already exists for database ID {id_database}.")
                    continue
                new_schema = SchemaInfo(
                    name=name,
                    id_database=id_database
                )                
                session.add(new_schema)
            session.commit()
            print(f"Schemas added successfully for database ID {id_database}.")
        except IntegrityError as ie:
            session.rollback()
            print(f"Integrity Error while adding schemas: {ie.orig}")
        except Exception as e:
            session.rollback()
            print(f"An error occurred while adding schemas: {e}")
        finally:
            session.close()
            
    def add_table(self, tables, id_schema, id_database, id_agent):
        session = self.Session()
        try:
            for table in tables:
                name = table["table_name"]
                existing_table = session.query(TableInfo).filter_by(name=name, id_schema=id_schema).first()
                if existing_table:
                    logging.info(f"Table '{name}' already exists for schema ID {id_schema}.")
                    continue
                
                new_table = TableInfo(
                    name=name,
                    id_schema=id_schema,
                    id_database=id_database,
                    id_agent=id_agent,
                    schema_name=table["schema_name"],
                    database_name=table["database_name"],
                    agent_name=table["agent_name"],
                    total_columns=table["total_columns"],
                    total_rows=table["total_rows"],
                    table_size=table["table_size"]
                )
                session.add(new_table)
            
            session.commit()
            logging.info(f"Tables added successfully for schema ID {id_schema}.")
        except IntegrityError as ie:
            session.rollback()
            logging.error(f"Integrity Error while adding tables: {ie.orig}")
        except Exception as e:
            session.rollback()
            logging.error(f"An error occurred while adding tables: {e}")
        finally:
            session.close()
            
    def delete_agent(self, agent_id):
        session = self.Session()
        try:
            agent = session.query(AgentInfo).get(agent_id)
            if agent:
                session.delete(agent)
                session.commit()
                print(f"Agent with ID {agent_id} deleted successfully.")
            else:
                print(f"Agent with ID {agent_id} does not exist.")
        except Exception as e:
            session.rollback()
            print(f"An error occurred while deleting agent: {e}")
        finally:
            session.close()
            
    def delete_database(self, database_id):
        session = self.Session()
        try:
            database = session.query(DatabaseInfo).get(database_id)
            if database:
                session.delete(database)
                session.commit()
                print(f"Database with ID {database_id} deleted successfully.")
            else:
                print(f"Database with ID {database_id} does not exist.")
        except Exception as e:
            session.rollback()
            print(f"An error occurred while deleting database: {e}")
        finally:
            session.close()
            
    def get_id_database_by_instance(self, db_instance):
        session = self.Session()
        try:
            db_info = session.query(DatabaseInfo).filter_by(instance=db_instance).first()
            if db_info:
                return db_info.id
            else:
                print(f"No database record found for instance: {db_instance}")
                return None
        except Exception as e:
            print(f"An error occurred while retrieving databases: {e}")
        finally:
            session.close()
            
    def get_id_schema_by_name(self, schema_name):
        session = self.Session()
        try:
            schema_info = session.query(SchemaInfo).filter_by(name=schema_name).first()
            if schema_info:
                return schema_info.id
            else:
                print(f"No schema record found for schema: {schema_name}")
                return None
        except Exception as e:
            print(f"An error occurred while retrieving schemas: {e}")
        finally:
            session.close()
            
    def get_id_agent_from_database(self, id_database):
        session = self.Session()
        try:
            db_info = session.query(DatabaseInfo).filter_by(id=id_database).first()
            if db_info:
                return db_info.id_agent, 
            else:
                print(f"No id_agent record found for id_database: {id_database}")
                return None
        except Exception as e:
            print(f"An error occurred while retrieving schemas: {e}")
        finally:
            session.close()
            
    def get_agent_from_id_agent(self, id_agent):
        session = self.Session()
        try:
            agent_info = session.query(AgentInfo).filter_by(id=id_agent).first()
            if agent_info:
                return agent_info.name
            else:
                print(f"No agent record found for id_agent: {id_agent}")
                return None
        except Exception as e:
            print(f"An error occurred while retrieving data: {e}")
        finally:
            session.close()

    def get_database_from_id_database(self, id_database):
        session = self.Session()
        try:
            db_info = session.query(DatabaseInfo).filter_by(id=id_database).first()
            if db_info:
                return db_info.instance
            else:
                print(f"No agent record found for id_database: {id_database}")
                return None
        except Exception as e:
            print(f"An error occurred while retrieving data: {e}")
        finally:
            session.close()

    def collect_and_add_schemas(self, database_type, source_database, source_user, source_host, source_port, source_password):
        external_db_service = ExternalDatabaseService()
        schemas = external_db_service.get_database_schemas(
            database_type, 
            source_database, 
            source_user, 
            source_host, 
            source_port, 
            source_password
        )
                
        id_database = self.get_id_database_by_instance(source_database)
        
        if id_database is None:
            print(f"Cannot proceed without a valid id_database for db_instance: {source_database}")
            return
        
        self.add_schema(schemas, id_database)
        
    def collect_and_add_tables(self, database_type, source_database, source_user, source_host, source_port, source_password):
        external_db_service = ExternalDatabaseService()
        tables = external_db_service.get_database_table(
            database_type, 
            source_database, 
            source_user, 
            source_host, 
            source_port, 
            source_password
        )
        
        data_service = DataService(self.engine)
        
        for table in tables:
            schema_name = table["schema_name"]
            id_schema = data_service.get_id_schema_by_name(schema_name)
            id_database = data_service.get_id_database_by_instance(source_database)
            database_name = data_service.get_database_from_id_database(id_database)
            id_agent = data_service.get_id_agent_from_database(id_database)
            agent_name = data_service.get_agent_from_id_agent(id_agent)
            
            if id_schema is None or id_database is None or id_agent is None:
                logging.error(f"Cannot proceed without valid ids for schema '{schema_name}', database '{self.source_database}'.")
                continue
            
            data_service.add_table([table], id_schema, id_database, id_agent)
        
            
class ExternalDatabaseService:
    def __init__(self):
        pass

    def get_database_schemas(self, database_type, database, user, host, port, password):        
        if database_type == "postgresql":    
            database_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
            query = text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
                ORDER BY schema_name
            """)

        elif database_type == "mysql":
            database_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            query = text("""
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
            """)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")

        engine = create_engine(database_url)
        schemas = []

        try:
            with engine.connect() as connection:
                result = connection.execute(query)
                schemas = [{"schema_name": row["schema_name"], "db_instance":database} for row in result]
        except SQLAlchemyError as e:
            logging.error(f"An error occurred while retrieving schemas: {e}")
        finally:
            engine.dispose()

        return schemas
    
    def get_database_table(self, database_type, database, user, host, port, password):
        if database_type == "postgresql":
            database_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
            schema_query = text("""
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema','pg_catalog')
                ORDER BY schema_name;
            """)
            
            table_query = text(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema_name
                ORDER BY table_name;
            """)
            total_columns_query = text(f"""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = :schema_name AND table_name = :table_name;
            """)
            table_size_query = text("""SELECT pg_total_relation_size('"' || :schema_name || '"."' || :table_name || '"') AS table_size;
            """)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
            
        engine = create_engine(database_url)
        tables = []
        
        try:
            with engine.connect() as connection:
                schema_results = connection.execute(schema_query)
                schemas = [row['schema_name'] for row in schema_results]
                
                for schema_name in schemas:
                    table_results = connection.execute(table_query, {'schema_name':schema_name})

                    for row in table_results:
                        table_name = row["table_name"]

                        total_columns_result = connection.execute(total_columns_query, {"schema_name":schema_name, "table_name":table_name})
                        total_columns = total_columns_result.scalar()

                        total_rows_result = connection.execute(text(f"SELECT COUNT(*) FROM {schema_name}.{table_name}"))
                        total_rows = total_rows_result.scalar()

                        table_size_result = connection.execute(table_size_query, {"schema_name":schema_name, "table_name":table_name})
                        table_size = table_size_result.scalar()

                        tables.append({
                            "table_name":table_name,
                            "total_columns":total_columns,
                            "total_rows":total_rows,
                            "table_size":table_size,
                            "schema_name":schema_name,
                            "database_name":database
                        })
                        
        except SQLAlchemyError as e:
            logging.error(f"An error occurred while retrieving table information: {e}")
        finally:
            engine.dispose()
        
        return tables
    
class DataQualityEvaluator:
    def __init__(self, engine):
        self.engine = engine

    def table_list(self):
        df_db_info = pd.read_sql("""SELECT * FROM aidm.db_info;""", self.engine)
        df_schema_info = pd.read_sql("""SELECT * FROM aidm.schema_info""", self.engine)
        df_table_info = pd.read_sql("""SELECT * FROM aidm.table_info""", self.engine)

        df = df_db_info[["id","instance"]].merge(df_schema_info[["id_db","id","name"]], left_on="id", right_on="id_db", how="left")
        df = df.drop("id_db", axis=1)
        df.columns = ["id_db","database", "id_schema", "schema"]
        df = df[df["schema"]!="pg_toast"].merge(df_table_info[["id_schema", "id", "name"]], on="id_schema", how="left")
        df.columns = ["id_db","database","id_schema","schema","id_table","table"]
        return df

    def completeness_column(self, series):
        row_total = series.shape[0]
        row_null = series.isna().sum()
        row_complete = row_total - row_null
        complete_perc = round(100 * row_complete / row_total, 2) if row_total != 0 else 0
        df_result = pd.DataFrame({"row_total": [row_total], "row_null": [row_null], "row_complete": [row_complete], "complete_perc": [complete_perc]})
        return df_result

    def completeness_table(self, table):
        column_list = table.columns.to_list()
        df_result = pd.DataFrame({"column": pd.Series(dtype=str), "row_total": pd.Series(dtype=int), "row_null": pd.Series(dtype=int), "row_complete": pd.Series(dtype=int), "complete_perc": pd.Series(dtype=float)})
        for column in column_list:
            df_column = self.completeness_column(table[column])
            df_column["column"] = column
            df_result = pd.concat([df_result, df_column], axis=0)
        return df_result

    def completeness_table_all(self):
        df_table_all = self.table_list()
        df_result = pd.DataFrame({
            "schema": pd.Series(dtype=str), "table": pd.Series(dtype=str), "column": pd.Series(dtype=str),
            "row_total": pd.Series(dtype=int), "row_null": pd.Series(dtype=int), "row_complete": pd.Series(dtype=int), "complete_perc": pd.Series(dtype=float)})
        for i in range(df_table_all.shape[0]):
            schema = df_table_all["schema"][i]
            table = df_table_all["table"][i]
            df_table = pd.read_sql(f"""SELECT * FROM {schema}.{table}""", self.engine)
            df_table_result = self.completeness_table(df_table)
            df_table_result["schema"] = schema
            df_table_result["table"] = table
            df_result = pd.concat([df_result, df_table_result], axis=0)
        df_table_all = df_table_all.merge(df_result, on=["schema", "table"], how="left")
        return df_table_all

    def uniqueness_column(self, series):
        series = series.dropna()
        row_total = series.shape[0]
        try:
            value_unique = series.nunique()
            unique_perc = round(100 * value_unique / row_total, 2) if row_total != 0 else 0
            df_result = pd.DataFrame({"row_total": [row_total], "value_unique": [value_unique], "unique_perc": [unique_perc]})
        except TypeError:
            df_result = pd.DataFrame({"row_total": [row_total], "value_unique": [0], "unique_perc": [0.0]})
        return df_result

    def uniqueness_table(self, table):
        column_list = table.columns.to_list()
        df_result = pd.DataFrame({"column": pd.Series(dtype=str), "row_total": pd.Series(dtype=int), "value_unique": pd.Series(dtype=int), "unique_perc": pd.Series(dtype=float)})
        for column in column_list:
            df_column = self.uniqueness_column(table[column])
            df_column["column"] = column
            df_result = pd.concat([df_result, df_column], axis=0)
        return df_result

    def uniqueness_table_all(self):
        df_table_all = self.table_list()
        df_result = pd.DataFrame({
            "schema": pd.Series(dtype=str), "table": pd.Series(dtype=str), "column": pd.Series(dtype=str),
            "row_total": pd.Series(dtype=int), "value_unique": pd.Series(dtype=int), "unique_perc": pd.Series(dtype=float)})
        for i in range(df_table_all.shape[0]):
            schema = df_table_all["schema"][i]
            table = df_table_all["table"][i]
            df_table = pd.read_sql(f"""SELECT * FROM {schema}.{table}""", self.engine)
            df_table_result = self.uniqueness_table(df_table)
            df_table_result["schema"] = schema
            df_table_result["table"] = table
            df_result = pd.concat([df_result, df_table_result], axis=0)
        df_table_all = df_table_all.merge(df_result, on=["schema", "table"], how="left")
        return df_table_all

    def validity_column(self, series, data):
        series = series.dropna()
        row_total = series.shape[0]

        if data == "email":
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        elif data == "phone":
            pattern = r"^(\+62|62|0)8[1-9][0-9]{6,10}$"
        else:
            pass
        
        df = pd.DataFrame({"value":series})
        df["is_valid"] = df.iloc[:, 0].apply(lambda x: bool(re.match(pattern, str(x))))
        value_valid = df["is_valid"].sum()
        valid_perc = round(100 * value_valid/row_total, 2) if row_total !=0 else 0

        df_result = pd.DataFrame({"row_total":[row_total], "value_unique":[value_valid], "valid_perc":[valid_perc]})
        return df

    def redundancy_table(self, df):
        df = df.dropna()
        row_total = df.shape[0]
        row_duplicates = df.duplicated().sum()
        redundancy_perc = round(100 * row_duplicates/row_total, 2) if row_total != 0 else 0

        df_result = pd.DataFrame({"row_total":[row_total], "row_duplicates":[row_duplicates], "redundancy_perc":[redundancy_perc]})
        return df_result

    def timeliness_table(self, df, update_time_column):
        df = df.dropna()
        df.loc[:, update_time_column] = df[update_time_column].dt.date
        unique_dates = df[update_time_column].drop_duplicates().sort_values()
        time_differences = unique_dates.diff().dt.days
        average_time_distance = round(time_differences.mean(),2)

        df_result = pd.DataFrame({"update_time_column":[update_time_column], "average_update_time":[average_time_distance]})    
        return df_result
    
    def integrity_column(self, df, column_name, df_reference, column_reference_name):
        list_existing = df[column_name].dropna().to_list() 
        row_total = len(list_existing)
        list_reference = list(df_reference[column_reference_name].unique())
        value_integrous = sum(elem in list_reference for elem in list_existing)
        integrity_perc = round(100 * value_integrous/row_total, 2) if row_total != 0 else 0

        df_result = pd.DataFrame({"row_total":[row_total], "column_name":[column_name], "column_reference_name":[column_reference_name], "value_integrous":[value_integrous], "integrity_perc":[integrity_perc]})

        return df_result

    def consistency_column(self, df, column_name, left_join_key, df_reference, column_reference_name, right_join_key):
        df = df[[left_join_key, column_name]].dropna(subset=[column_name])
        row_total = df.shape[0]

        df_reference = df_reference[[right_join_key, column_reference_name]]
        df_join = df.merge(df_reference, left_on=left_join_key, right_on=right_join_key, how="left")
        # df_result = pd.DataFrame({"row_total":[row_total], "column_name":[column_name], "column_reference_name":[column_reference_name], "value_integrous":[value_integrous], "integrity_perc":[integrity_perc]})

        try:
            df_join["is_consistence"] = df_join[column_name] == df_join[column_reference_name]
        except KeyError:
            df_join["is_consistence"] = df_join[f"{column_name}_x"] == df_join[f"{column_name}_y"]

        value_consistency = df_join["is_consistence"].sum()
        consistency_perc = round(100 * value_consistency/row_total, 2) if row_total != 0 else 0

        df_result = pd.DataFrame({"row_total":[row_total], "column_name":[column_name], "column_reference_name":[column_reference_name], "value_consistency":[value_consistency], "consistency_perc":[consistency_perc]})

        return df_result

    def masking_column(self, df, column_name, mask_chars=['*', 'X', '#', '@', '?', '-', '+']):
        df = df[[column_name]].dropna()
        row_total = df.shape[0]

        mask_char_pattern = "[" + re.escape("".join(mask_chars)) + "]"

        def contains_masking_chars(value):
            return bool(re.search(mask_char_pattern, value))

        df["is_masked"] = df[column_name].apply(contains_masking_chars)
        value_masked = df["is_masked"].sum()
        masked_perc = round(100 * value_masked/row_total, 2) if row_total != 0 else 0
        df_result = pd.DataFrame({"row_total":[row_total], "column":[column_name], "value_masked":[value_masked], "masked_perc":[masked_perc]})

        return df_result
    
    def primary_key_table(self):
        df = self.table_list()
        df_pk = pd.read_sql("""SELECT * FROM aidm.table_pk_fk""", self.engine)[["id_db","id_schema","table_name","is_primary_key","primary_key"]]
        df_pk = df_pk.rename(columns={"table_name":"table"})
        df = df.merge(df_pk, on=["id_db","id_schema","table"], how="left")
        return df

    def primary_key_schema(self):
        df = self.primary_key_table()
        df = df.groupby(["id_db","database","id_schema","schema"]).agg({"is_primary_key":"mean"}).reset_index()
        df = df.rename(columns={"is_primary_key":"primary_key_perc"})
        df["primary_key_perc"] = df["primary_key_perc"] * 100
        return df
    
    def calculate_brace_series(self, brace):
        brace_to_list = [int(num) for num in re.findall(r'\d+', brace)]
        n_foreign_key = len(brace_to_list)
        valid_foreign_key = sum(brace_to_list)
        foreign_key_perc = round(100 * valid_foreign_key/n_foreign_key, 2) if n_foreign_key != 0 else 0
        return foreign_key_perc

    def foreign_key_table(self):
        df = self.table_list()
        df_pk = pd.read_sql("""SELECT * FROM aidm.table_pk_fk""", self.engine)[["id_db","id_schema","table_name","foreign_key","is_foreign_key_valid"]]
        df_pk = df_pk.rename(columns={"table_name":"table"})
        df = df.merge(df_pk, on=["id_db","id_schema","table"], how="left")
        df["foreign_key_perc"] = df["is_foreign_key_valid"].apply(lambda x: self.calculate_brace_series(x))
        return df

    def foreign_key_schema(self):
        df = self.foreign_key_table()
        df = df.groupby(["id_db","database","id_schema","schema"]).agg({"foreign_key_perc":"mean"}).reset_index()
        df["foreign_key_perc"] = df["foreign_key_perc"].round(2) 
        return df
    
    def index_table(self):
        df = self.table_list()
        df_index = pd.read_sql("""SELECT * FROM aidm.table_index""", self.engine)[["id_db","id_schema","table_name","is_indexed","index"]]
        df_index = df_index.rename(columns={"table_name":"table"})
        df = df.merge(df_index, on=["id_db","id_schema","table"], how="left")
        return df

    def index_schema(self):
        df = self.index_table()
        df = df.groupby(["id_db","database","id_schema","schema"]).agg({"is_indexed":"mean"}).reset_index()
        df = df.rename(columns={"is_indexed":"index_perc"})
        df["index_perc"] = 100 * df["index_perc"].round(2)
        return df
    
    def comment_table(self):
        df = self.table_list()
        df_comment = pd.read_sql("""SELECT * FROM aidm.table_comment""", self.engine)[["id_db","id_schema","table_name","is_commented","comment"]]
        df_comment = df_comment.rename(columns={"table_name":"table"})
        df = df.merge(df_comment, on=["id_db","id_schema","table"], how="left")
        return df

    def comment_schema(self):
        df = self.comment_table()
        df = df.groupby(["id_db","database","id_schema","schema"]).agg({"is_commented":"mean"}).reset_index()
        df = df.rename(columns={"is_commented":"comment_perc"})
        df["comment_perc"] = 100 * df["comment_perc"].round(2)
        return df
    
    # def get_percentile(df, level="database", perc=(20, 40, 60, 80)):
    #     if level == "database":
    #         db_list = list(df["id_db"].unique())
    #         df_result = pd.DataFrame({"id_db":pd.Series(dtype=int), "threshold":pd.Series(dtype=object)})
    #         for db in db_list:
    #             df_filter = df[df["id_db"]==db]
    #             complete_list = df_filter["complete_perc"]
    #             threshold_dict = {
    #                 "0":min(complete_list),
    #                 "1":np.percentile(complete_list, perc[0]),
    #                 "2":np.percentile(complete_list, perc[1]),
    #                 "3":np.percentile(complete_list, perc[2]),
    #                 "4":np.percentile(complete_list, perc[3]),
    #                 "5":max(complete_list),
    #             }
    #             df_db = pd.DataFrame({"id_db":[db], "threshold":[threshold_dict]})
    #             df_result = pd.concat([df_result, df_db], axis=0)
    #         df_output = df.merge(df_result, on="id_db", how="left")
    #     elif level == "schema":
    #         schema_list = list(df[df["schema"]!="pg_toast"]["id_schema"].unique())
    #         df_result = pd.DataFrame({"id_schema":pd.Series(dtype=int), "threshold":pd.Series(dtype=object)})
    #         for schema in schema_list:
    #             df_filter = df[df["id_schema"]==schema]
    #             complete_list = df_filter["complete_perc"]
    #             threshold_dict = {
    #                 "0":min(complete_list),
    #                 "1":np.percentile(complete_list, perc[0]),
    #                 "2":np.percentile(complete_list, perc[1]),
    #                 "3":np.percentile(complete_list, perc[2]),
    #                 "4":np.percentile(complete_list, perc[3]),
    #                 "5":max(complete_list),
    #             }
    #             df_schema = pd.DataFrame({"id_schema":[schema], "threshold":[threshold_dict]})
    #             df_result = pd.concat([df_result, df_schema], axis=0)
    #         df_output = df.merge(df_result, on="id_schema", how="left")
    #     return df_output