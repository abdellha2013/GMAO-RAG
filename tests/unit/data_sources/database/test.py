from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from  app.data_sources import DataSourceOrchestrator

orchestrator = DataSourceOrchestrator()
document = orchestrator.load({
"driver": "mysql",
"host": "localhost",
"database": "gmao_rag_test",
"user": "root",
"password": "Zzdv6401",
"table": "equipements",})

if document.source_path :
    print("the source_path of the file is : ",document.source_type)
else :
    print("the source_path not existe .")



if document.extension =="." :
    print("the extension of the file is : ",document.extension)
else :
    print("the extension not existe .")




