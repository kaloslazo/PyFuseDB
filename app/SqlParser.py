class SqlParser:
    def __init__(self):
        self.query = ""
        print("Parser SQL inicializado.")

    def parseQuery(self, query):
        query = query.lower()
        parts = query.split()
        select_index = parts.index('select')
        from_index = parts.index('from')
        like_index = parts.index('like') if 'like' in parts else len(parts)
        fields = ' '.join(parts[select_index+1:from_index]).split(',')
        fields = [field.strip() for field in fields]
        table = parts[from_index+1:like_index][0] if from_index+1 < like_index else None
        like_term = ' '.join(parts[like_index+1:]) if like_index < len(parts) else None
        return {
            'fields': fields,
            'table': table,
            'like_term': like_term
        }

    def selectQuery(self, querySelect):
        attributesForSelect = []

        print(f"Procesando query SELECT: {querySelect}")
        for i in range(1, len(querySelect)):
            if querySelect[i].lower() == "from":
                break
            attributesForSelect.extend(querySelect[i].replace(',', '').split())
        print(f"Atributos para SELECT: {attributesForSelect}")

        return attributesForSelect

    def fromQuery(self, queryFrom):
        print(f"Procesando query FROM: {queryFrom}")
        tablesForFrom = []

        for i in range(len(queryFrom)):
            if queryFrom[i].lower() == "from":
                if i + 1 < len(queryFrom):
                    tablesForFrom.append(queryFrom[i + 1])
                break

        return tablesForFrom
    
    def likeQuery(self, queryLike):
        print(f"Procesando query LIKE: {queryLike}")
        like_clause = []
        if "like" in queryLike:
            like_index = queryLike.index("like")
            like_clause = queryLike[like_index:]
        return like_clause
