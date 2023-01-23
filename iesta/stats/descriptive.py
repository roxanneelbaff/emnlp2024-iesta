# %%
from typing import List
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
#By default, however, the normalization is applied to the entire distribution, so this simply rescales the height of the bars. 
# #By setting common_norm=False, each subset will be normalized independently:
#sns.pairplot(style_features_df[mpqa_cols+["effect"]], hue="effect")
_type: str= "boxplot"# "normalized_histo" "swarmplot" "boxplot" 
def plot_distribution(df:pd.DataFrame, cols: List[str], _type:str= "normalized_histo"):
    for col in cols:
        if _type == "normalized_histo":
            sns.displot(df, x=col, hue="effect", 
                        multiple="dodge",  stat="probability", 
                        common_norm=False)
        elif _type=="swarmplot":
            sns.stripplot(data=df, x="effect", y=col)
        elif _type == "boxplot":
            sns.catplot(data=df, x="effect", y=col, kind="box")
                        
        #sns.displot(style_features_df, x=mpqa_col, hue="effect", kind="kde", fill=True)
        plt.title(f"{(col.split('_')[0]).capitalize()} {(col.split('_')[-2]).capitalize()}")
        plt.show()
        
    