# import wandb
# import pandas as pd

# api = wandb.Api()

# runs = {
#     "ours": "ducthu2003/TARGET/to7336gy",
#     "finetune": "ducthu2003/TARGET/qoue8ueb",
#     "lwf": "ducthu2003/TARGET/tjsya4ed",
#     "ewc": "ducthu2003/TARGET/xhg3yluy",
#     "icarl": "ducthu2003/TARGET/57cahmi0",
# }
# task_key = "Task_4, accuracy"

# dfs = []

# for model_name, run_id in runs.items():

#     print(f"Downloading {model_name}")

#     run = api.run(run_id)

#     rows = []

#     for row in run.scan_history():
#         if task_key in row:
#             rows.append({
#                 "_step": row["_step"],
#                 model_name: row[task_key]
#             })

#     df = pd.DataFrame(rows)

#     dfs.append(df)

# # merge theo step
# final_df = dfs[0]

# for df in dfs[1:]:
#     final_df = pd.merge(final_df, df, on="_step", how="outer")

# # sort step
# final_df = final_df.sort_values("_step")
# # bỏ dòng không có accuracy
# final_df = final_df.dropna(subset=["ours","finetune","lwf","ewc","icarl"], how="all")
# # reset index
# final_df = final_df.reset_index(drop=True)

# print("\nPreview:")
# print(final_df.head())

# final_df.to_csv("Task4_accuracy_all_models.csv", index=False)

# print("\nSaved to Task4_accuracy_all_models.csv")


import wandb
import pandas as pd

api = wandb.Api()
run = api.run("ducthu2003/TARGET/ccdok8rl")

history = run.history(keys=[ "Task_1_acc"])
df = pd.DataFrame(history)
print(df.head())

df.to_csv("Task_1_acc.csv", index=False)
print("Saved to Task_1_acc.csv")

