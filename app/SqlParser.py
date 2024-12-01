import re

class SqlParser:
    def __init__(self):
        self.query = ""
        print("Parser SQL inicializado.")

    def parseQuery(self, query):
        """Parsea una consulta SQL con soporte para LIKE y LIKETO"""
        # Convertir a minúsculas para consistencia
        query = query.lower()
        parts = query.split()
        
        try:
            # Encontrar las partes principales de la consulta
            select_index = parts.index('select')
            from_index = parts.index('from')
            
            # Buscar LIKE o LIKETO
            like_index = -1
            like_term = None
            
            for i, part in enumerate(parts):
                if part in ['like', 'liketo']:
                    like_index = i
                    # Extraer el término después de LIKE/LIKETO hasta LIMIT si existe
                    limit_index = len(parts)
                    if 'limit' in parts:
                        limit_index = parts.index('limit')
                    like_term = ' '.join(parts[like_index + 1:limit_index])
                    break
            
            # Extraer campos
            fields = ' '.join(parts[select_index + 1:from_index]).split(',')
            fields = [field.strip() for field in fields]
            
            # Extraer tabla
            table = parts[from_index + 1:like_index] if like_index > -1 else parts[from_index + 1]
            table = table[0] if isinstance(table, list) else table
            
            return {
                'fields': fields,
                'table': table,
                'like_term': like_term
            }
            
        except ValueError as e:
            print(f"Error parseando la consulta: {str(e)}")
            # Valores por defecto si hay error
            return {
                'fields': ['*'],
                'table': 'songs',
                'like_term': None
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
    
    def parseQueryPostgres(self, query):
        # Convertimos todo a minúsculas y separamos por espacios
        query = query.lower()
        parts = query.split()

        # Buscamos las posiciones de los componentes clave
        select_index = parts.index("select")
        from_index = parts.index("from")
        like_index = parts.index("like") if "like" in parts else len(parts)

        # Extraemos los campos que se seleccionan (entre SELECT y FROM)
        fields = ' '.join(parts[select_index + 1:from_index]).split(',')
        fields = [field.strip() for field in fields]

        # Extraemos la tabla (entre FROM y LIKE)
        table = parts[from_index + 1:like_index][0] if from_index + 1 < like_index else None

        # Extraemos el término de LIKE (después de LIKE)
        like_term = ' '.join(parts[like_index + 1:]) if like_index < len(parts) else None

        # Si like_term no es None, lo separamos por espacios y construimos una consulta con el operador "&"
        if like_term:
            tokens = like_term.split()
            # Unimos los tokens con "&" para que la consulta busque todos los términos juntos
            like_query = ' & '.join(tokens)
        else:
            like_query = ""

        # Construimos la consulta SQL con búsqueda de texto completo
        sql_query = f"SELECT {', '.join(fields)} FROM {table} WHERE to_tsvector('english', texto_concatenado) @@ to_tsquery('english', '{like_query}')"

        return sql_query
