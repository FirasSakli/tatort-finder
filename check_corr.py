import duckdb

con = duckdb.connect("data/processed/tatort.duckdb")
df = con.execute(
    "SELECT d.death_score, p.plot_len "
    "FROM death_features d JOIN plots p USING(Folge) "
    "WHERE p.plot IS NOT NULL"
).fetchdf()
con.close()

corr = df["death_score"].corr(df["plot_len"])
print(f"Correlation death_score vs plot length: {corr:.2f}")