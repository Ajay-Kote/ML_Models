import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


class SHAPExplainer:

    def __init__(
        self,
        model_path="models/saved_model.pkl",
        data_path="data/processed/features.csv",
        sample_size=1000
    ):

        print("Loading model...")

        self.model = joblib.load(model_path)

        print("Loading dataset...")

        df = pd.read_csv(data_path)

        self.X = df.drop(
            columns=["URL", "Domain", "TLD", "Label"],
            errors="ignore"
        )

        if len(self.X) > sample_size:

            self.X = self.X.sample(
                n=sample_size,
                random_state=42
            )

        print(f"Using {len(self.X)} samples for SHAP")

        self.explainer = shap.TreeExplainer(self.model)

        os.makedirs("results/shap", exist_ok=True)

    def compute(self):

        print("Computing SHAP values...")

        shap_values = self.explainer(self.X)

        return shap_values

    def summary_plot(self, shap_values):

        print("Generating Summary Plot...")

        plt.figure(figsize=(10, 6))

        shap.plots.beeswarm(
            shap_values,
            max_display=20,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            "results/shap/shap_summary.png",
            dpi=300
        )

        plt.close()

        print("Saved -> shap_summary.png")

    def bar_plot(self, shap_values):

        print("Generating Bar Plot...")

        plt.figure(figsize=(10, 6))

        shap.plots.bar(
            shap_values,
            max_display=20,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            "results/shap/shap_bar.png",
            dpi=300
        )

        plt.close()

        print("Saved -> shap_bar.png")

    def waterfall_plot(self, shap_values, index=0):

        print("Generating Waterfall Plot...")

        plt.figure(figsize=(8, 6))

        shap.plots.waterfall(
            shap_values[index],
            max_display=15,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            "results/shap/shap_waterfall.png",
            dpi=300
        )

        plt.close()

        print("Saved -> shap_waterfall.png")

    def top_features(self, shap_values, index=0):

        print("\nTop Features\n")

        values = shap_values[index].values

        names = shap_values.feature_names

        result = list(zip(names, values))

        result = sorted(
            result,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature, value in result[:10]:

            direction = "↑" if value > 0 else "↓"

            print(
                f"{feature:<30} {direction} {value:.4f}"
            )


if __name__ == "__main__":

    explainer = SHAPExplainer(
        sample_size=1000
    )

    shap_values = explainer.compute()

    explainer.summary_plot(shap_values)

    explainer.bar_plot(shap_values)

    explainer.waterfall_plot(shap_values)

    explainer.top_features(shap_values)

    print("\nCompleted Successfully!")