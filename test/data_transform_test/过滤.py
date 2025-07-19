#过滤后得preds是[['tonnage (grt)', 'name', 'fate']]，col_dict[str(g_eid)]["output"]是[['name', 'fate']]
# 过滤后得preds是[]，col_dict[str(g_eid)]["output"]是[['year', 'position', 'event']]
# ['event title', 'arena']
# 过滤后得preds是[['event title', 'arena']]，col_dict[str(g_eid)]["output"]是[['date', 'event title', 'arena', 'location']]
# 过滤后得preds是[]，col_dict[str(g_eid)]["output"]是[['year', 'album', 'single']]
# preds=[]
# t=[['year', 'position', 'event']]
# all_preds = preds + t
# pred = set().union(*all_preds)
# breakpoint()
# print()
for idx, n in enumerate(range(1)):
    print(f'这是第{idx+1}次到这里')