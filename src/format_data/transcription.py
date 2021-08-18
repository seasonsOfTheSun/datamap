import cmapPy.pandasGEXpress.parse
import pandas

data = cmapPy.pandasGEXpress.parse.parse_gctx.parse("data/raw/LDS-1194/Data/zspc_n70323x22268.gctx")

features = data.data_df.transpose()



landmark_features = data.row_metadata_df.query("pr_is_lmark == 'Y'").index

assert not data.row_metadata_df.loc[landmark_features].pr_gene_symbol.duplicated().any()

features = features[landmark_features]

features.columns = data.row_metadata_df.loc[landmark_features].pr_gene_symbol.values

features = features/features.std()

assert not features.isna().any().any()



data.col_metadata_df.columns


names = data.col_metadata_df.pert_desc

metadata = data.col_metadata_df
metadata["NAME"] = data.col_metadata_df.pert_desc

features.to_csv("data/intermediate/transcriptional/features.csv")
metadata.to_csv("data/intermediate/transcriptional/metadata.csv")
pandas.Series(names).to_csv("data/intermediate/transcriptional/drug_names.csv", header = None)
