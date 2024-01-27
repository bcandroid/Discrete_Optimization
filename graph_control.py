import networkx as nx

# Graph oluştur
G = nx.Graph()

# Dosyayı oku ve kenarları ekle
with open('input.txt', 'r') as input_file:
    lines = input_file.readlines()

n, e = map(int, lines[0].split())

for line in lines[1:]:
    u, v = map(int, line.split())
    G.add_edge(u, v)

# Renklendirmeyi yap
coloring = nx.coloring.greedy_color(G, strategy="largest_first")

sorted_nodes = sorted(coloring.keys())

# Düğüm renklerini sırayla bastır
#print(sorted_nodes)
for node in sorted_nodes:
    color = coloring[node]
    #print(color)

# Toplam renk sayısını bastır
num_colors = len(set(coloring.values()))
print("Total number of colors:", num_colors)
