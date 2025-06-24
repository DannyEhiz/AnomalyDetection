from tinydb import TinyDB, Query, where 
import os 
import json


def retrieveRecord(fileName: str, tableName: str, itemListName: any) -> list | None:
    """
    Retrieve a list of items from a document inside a TinyDB table.

    Args:
        fileName (str): Path to the TinyDB database file.
        tableName (str): Name of the table to query.
        itemListName (str): Name field of the document to retrieve.

    Returns:
        List or None: List of items if document exists, else None.
    """
    if not os.path.isfile(fileName):
        # raise FileNotFoundError(f"Database file '{fileName}' does not exist")
        return None
    db = TinyDB(fileName)
    try:
        if tableName not in db.tables():
            return None

        record = db.table(tableName).get(where('name') == itemListName)
        return record['value'] if record else None
    finally:
        db.close() 
    
def updateRecord(filename: str, tableName: str, itemListName: str, itemToAdd, append = True):
    """
    Append or replace items in the document's value list inside TinyDB.

    Args:
        filename (str): Path to TinyDB file.
        tableName (str): Name of the table.
        itemListName (str): Document identifier ('name' field value).
        itemToAdd: Item to add or set.
        append (bool): If True, append item to list. If False, replace list with item.

    Raises:
        Exception: If file or table does not exist.
    """
    if os.path.isfile(filename):
        db = TinyDB(filename)
        if tableName in db.tables():
            records = db.table(tableName).get(Query().name==itemListName)
            if records:
                records = records['value']
                if append:
                    # if isinstance(itemToAdd, str) or isinstance(itemToAdd, int):
                        if itemToAdd not in records:
                            records.append(itemToAdd)
                            db.table(tableName).update({'value': records}, where('name')==itemListName)
                        else:
                            pass # Item already exists, do nothing

                else:
                     db.table(tableName).update({'value': [itemToAdd]}, where('name')==itemListName)
            else:
                db.table(tableName).insert({'name': itemListName, 'value': [itemToAdd]})
        else:
            raise Exception('table does not exist')
    else:
        raise Exception('db_file does not exist')
    

def createRecord(fileName: str, tableName: str, itemListName: any):
    """
    Create a new document with an empty list under the specified table.

    Args:
        fileName (str): Path to the TinyDB database file.
        tableName (str): Name of the table to create the document in.
        itemListName (str): Name field of the document.

    Raises:
        Exception: If table already exists in the database.
    """
    db = TinyDB(fileName)
    if tableName in db.tables():
        raise Exception('Tablename already exist. Use Update function instead')
    else:
        db.table(tableName).insert({'name': itemListName, 'value':[]})


def tableIsExisting(fileName: str, tableName: str):
    """
    Check if a table already exists in the TinyDB database file.

    Args:
        fileName (str): Path to the TinyDB database file.
        tableName (str): Name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    if os.path.isfile(fileName):
        db = TinyDB(fileName)
        return True if tableName in db.tables() else False
    else:
        return False


def removeItemFromRecord(fileName, tableName, itemListName, itemToRemove):
    """
    Remove an item from the list stored in a document within a TinyDB table.

    Args:
        fileName (str): Path to the TinyDB database file.
        tableName (str): Name of the table containing the document.
        itemListName (str): Name field of the document.
        itemToRemove: Item to remove from the list.

    Side Effects:
        Prints success or failure messages to console.
    """
    if os.path.isfile(fileName):
        db = TinyDB(fileName)
        if tableName in db.tables():
            try:
                record = db.table(tableName).get(where('name')==itemListName)
                if record:
                    current_list = list(record['value'])
                    if itemToRemove in current_list:
                        current_list.remove(itemToRemove)
                        db.table(tableName).update({'value': current_list}, where('name') == itemListName)
                    else:   pass
                else:   pass
            finally:
                db.close()