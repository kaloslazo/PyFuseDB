class SqlParser:
    def __init__(self):
        self.query = ""
        print("Parser SQL inicializado.")

    def parseQuery(self, query):
        self.query = query.lower()
        queryParsed = self.query.split()

        if "select" in queryParsed:
            return self.selectQuery(queryParsed)
        else:
            raise ValueError("La query no es válida, consulte el manual de usuario.")

    def selectQuery(self, querySelect):
        print(f"Procesando query SELECT: {querySelect}")
        return querySelect
