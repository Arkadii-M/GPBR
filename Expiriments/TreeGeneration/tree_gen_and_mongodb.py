from operator import itemgetter
from pymongo import MongoClient, errors, ASCENDING

def value_cursor(mongo_cursor, *keys):
    for record in mongo_cursor:
        yield itemgetter(*keys)(record)

def tree_hash(tree): 
    # Tree hash function
    # Just concatenate all elements in list
    return ''.join(tree)

def create_record(expr,h):
    return {'expression':expr, 'height':h, 'tree_hash': tree_hash(expr) }

def gen_all_tree_stuct(binary_cursor, unary_cursor, terminal_symbol_str, prev_expr_cursor):
    """
    This fucntion generate all possible general tree structures (without substituting exact terminals)
    TODO: add more information
    """
    # Generate all possible tree structures adding unary operators
    for uop in value_cursor(unary_cursor.clone(), 'symbol'):
        for expr in value_cursor(prev_expr_cursor.clone(),'expression'):
            yield [uop,*expr]
    
    # Generate all possible tree structures adding binary operators
    # [Terminal] operator [expression1]
    # [Expression1] operator [Terminal]
    # [Expression1] operator [Expression2]
    for bop in value_cursor(binary_cursor.clone(), 'symbol'):
        for expr1 in value_cursor(prev_expr_cursor.clone(),'expression'):
            yield [bop, terminal_symbol_str, *expr1]
            yield [bop, *expr1, terminal_symbol_str]

            for expr2 in value_cursor(prev_expr_cursor.clone(),'expression'):
                yield [bop, *expr1, *expr2]

def gen_full_tree_stuct(binary_cursor, unary_cursor, terminal_symbol_str, prev_expr_cursor):
    """
    This fucntion generate all possible general tree structures (without substituting exact terminals)
    TODO: add more information
    """
    # Generate all possible tree structures adding unary operators
    for uop in value_cursor(unary_cursor.clone(), 'symbol'):
        for expr in value_cursor(prev_expr_cursor.clone(),'expression'):
            yield [uop,*expr]
    
    # Generate all possible full tree structures adding binary operators
    # [Expression1] operator [Expression2]
    for bop in value_cursor(binary_cursor.clone(), 'symbol'):
        for expr1 in value_cursor(prev_expr_cursor.clone(),'expression'):
            for expr2 in value_cursor(prev_expr_cursor.clone(),'expression'):
                yield [bop, *expr1, *expr2]



# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017")  # Replace with your MongoDB connection string


## Some configuration

MAX_H = 3
GEN_ALL = True

# Access the desired database and collection
db = client["TestExpressions"]
if GEN_ALL:
    expressions_collection = db["AllTreeExpressions"]
else:
    expressions_collection = db["FullTreeExpressions"]

op_collection = db["Operators"]

binary_cursor = op_collection.find({'num_args':2})
unary_cursor = op_collection.find({'num_args':1})
terminal_symbol = op_collection.find({'num_args':0})[0]['symbol']

agregate_by_height = [
    {"$group" : {"_id":"$height", "count":{"$sum":1}}},
    { "$sort" : { "_id" : 1 } }
    ]
heights_count = expressions_collection.aggregate(agregate_by_height)
heights_count = {int(el["_id"]):el["count"] for el in heights_count }

total_count = expressions_collection.count_documents({})
print(f''' 
    Current state information:
    Binary operators: {list(value_cursor(binary_cursor.clone(),'symbol'))}
    Unary operators: {list(value_cursor(unary_cursor.clone(),'symbol'))}
    Terminal symbol: {terminal_symbol}
    Number of expressions in database by height: {heights_count}
    Total records: {total_count}
''')

if total_count == 0:
    print('No expressions in database. Initialize...')

    print('Add initial expression (terminal)')
    expressions_collection.insert_one(create_record([terminal_symbol],0))
    heights_count[0]=1

    print('Create unique index on expression')
    expressions_collection.create_index('tree_hash', unique=True)



curr_h = max(heights_count.keys())

if GEN_ALL:
    print(f'Start generating all tree structures.\nMaximum height: {MAX_H}\nCurrent height: {curr_h}')
    for h in range(curr_h+1,MAX_H+1,1):
        print(f'Generating trees with height: {h}')
        expr_coursor = expressions_collection.find({'height':h-1})
        tree_gen = gen_all_tree_stuct(binary_cursor,unary_cursor,terminal_symbol,expr_coursor)
        for tree in tree_gen:
            try:
                expressions_collection.insert_one(
                    create_record(tree,h))
            except errors.DuplicateKeyError:
                print(f'Duplicate error.\nCurrent tree: {tree}')
else:
    print(f'Start generating full tree structures.\nMaximum height: {MAX_H}\nCurrent height: {curr_h}')
    for h in range(curr_h+1,MAX_H+1,1):
        print(f'Generating trees with height: {h}')
        expr_coursor = expressions_collection.find({'height':h-1})
        tree_gen = gen_full_tree_stuct(binary_cursor,unary_cursor,terminal_symbol,expr_coursor)
        for tree in tree_gen:
            try:
                expressions_collection.insert_one(
                    create_record(tree,h))
            except errors.DuplicateKeyError:
                print(f'Duplicate error.\nCurrent tree: {tree}')    
