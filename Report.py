import datetime

class ReportGenerator:
    def __init__(self, report_path="octopy_model_report.txt"):
        self.report_path = report_path

    def _format_metrics(self, metrics_dict):
        return "\n".join([f"  - {metric}: {value:.4f}" for metric, value in metrics_dict.items()])

    def _format_hyperparams(self, hyperparams):
        return "\n".join([f"  - {k}: {v}" for k, v in hyperparams.items()])

    def generate_report(
        self,
        models: list,  # list of dicts: [{'name': ..., 'metrics': {...}, 'hyperparams': {...}}]
        dataset_description: str = None,
        preprocessing_steps: list = None,
        issues: list = None,
        conclusion: str = None
    ):
        report_lines = []

        # Timestamp
        report_lines.append(f"📝 OCTOPY MODEL REPORT - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)

        # Dataset
        if dataset_description:
            report_lines.append("\n📊 Dataset Description:\n" + dataset_description)

        # Preprocessing
        if preprocessing_steps:
            report_lines.append("\n🧹 Preprocessing Steps:")
            for step in preprocessing_steps:
                report_lines.append(f"  - {step}")

        # Model(s) Reporting
        for idx, model_info in enumerate(models, 1):
            report_lines.append(f"\n🚀 Model {idx}: {model_info.get('name', 'Unnamed Model')}")
            report_lines.append("-" * 40)

            if model_info.get('metrics'):
                report_lines.append("📈 Metrics:")
                report_lines.append(self._format_metrics(model_info['metrics']))
            else:
                report_lines.append("📈 Metrics: Not provided")

            if model_info.get('hyperparams'):
                report_lines.append("\n⚙️ Hyperparameters:")
                report_lines.append(self._format_hyperparams(model_info['hyperparams']))

        # Issues / Observations
        if issues:
            report_lines.append("\n⚠️ Known Issues / Observations:")
            for issue in issues:
                report_lines.append(f"  - {issue}")

        # Conclusion / Recommendations
        if conclusion:
            report_lines.append("\n🧠 Conclusion / Notes:")
            report_lines.append(conclusion)

        # Save Report
        with open(self.report_path, "w") as file:
            file.write("\n".join(report_lines))

        print(f"\n✅ Report generated and saved as: {self.report_path}")
