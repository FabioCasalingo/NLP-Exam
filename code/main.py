"""
Main pipeline orchestrator for the debiasing project
Runs: baseline → training → evaluation → comparison → visualization
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from src.data_loader import DataLoader
from src.model_manager import ModelManager
from src.evaluation import StereoSetEvaluator
from src.training import MLMTrainer
from src.visualization import BiasVisualizer, visualize_all_results


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_step(step_num: int, step_name: str):
    print("\n" + "-" * 70)
    print(f"STEP {step_num}: {step_name}")
    print("-" * 70 + "\n")


def save_experiment_log(log_data: dict, output_path: str = "results/experiment_log.json"):
    """Save complete experiment log to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\nExperiment log saved: {output_path}")


def run_baseline_evaluation(
    evaluator: StereoSetEvaluator,
    manager: ModelManager,
    loader: DataLoader,
    stereoset_split: str = 'test',
    batch_size: int = 16,
    use_subset: bool = False,
    subset_size: int = 100
) -> dict:
    """
    Step 1: evaluate the original pretrained model
    Returns dict with baseline metrics
    """
    print_step(1, "BASELINE EVALUATION")

    # load stereoset data
    print("Loading StereoSet data...")
    stereoset_data = loader.load_stereoset(stereoset_split)

    # use smaller subset for quick testing if needed
    if use_subset:
        print(f"Using subset of {subset_size} examples for quick testing")
        stereoset_data['intrasentence'] = stereoset_data.get('intrasentence', [])[:subset_size]

    # load baseline RoBERTa model
    print("\nLoading baseline model (RoBERTa-base)...")
    model, tokenizer = manager.load_pretrained_roberta()

    # save it for future reference
    baseline_path = manager.save_model(
        model, tokenizer,
        save_path=str(manager.baseline_dir),
        save_name="original"
    )
    print(f"Baseline model saved: {baseline_path}")

    # run evaluation on stereoset
    print("\nEvaluating baseline model on StereoSet...")
    start_time = time.time()

    baseline_results = evaluator.evaluate_stereoset(
        model, tokenizer, stereoset_data, batch_size=batch_size
    )

    eval_time = time.time() - start_time
    print(f"Evaluation time: {eval_time/60:.1f} minutes")

    # save results
    evaluator.save_results(
        baseline_results,
        "results/baseline_results.json",
        "roberta-base-baseline"
    )

    # check some high-bias examples
    print("\nAnalyzing high-bias examples...")
    high_bias = evaluator.get_high_bias_examples(baseline_results, top_k=5)

    print("\nTop 3 high-bias examples:")
    for i, item in enumerate(high_bias[:3], 1):
        example = item['example']
        print(f"\n  [{i}] Bias type: {example['bias_type']}")
        print(f"      Context: {example.get('context','')[:60]}...")
        print(f"      Bias score: {item['bias_score']:.3f}")

    return baseline_results


def run_training(
    trainer: MLMTrainer,
    manager: ModelManager,
    loader: DataLoader,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 8,
    use_subset: bool = False,
    subset_size: int = 500
) -> tuple:
    """
    Step 2: fine-tune the model for debiasing
    Returns (debiased_model, tokenizer, training_info)
    """
    print_step(2, "FINE-TUNING FOR DEBIASING")

    # load debiasing corpus
    print("Loading debiasing corpus...")
    corpus = loader.load_debiasing_corpus(balance=True)

    train_texts = corpus['train']
    val_texts = corpus['val']

    # quick test mode uses smaller subset
    if use_subset:
        print(f"Using subset of {subset_size} train examples for quick testing")
        train_texts = train_texts[:subset_size]
        val_texts = val_texts[:max(1, int(subset_size * 0.1))]

    print(f"  Train examples: {len(train_texts)}")
    print(f"  Val examples: {len(val_texts)}")

    # load baseline to start training from
    print("\nLoading baseline model...")
    model, tokenizer = manager.load_model_from_path(
        str(manager.baseline_dir / "original")
    )

    # start fine-tuning
    print("\nStarting fine-tuning...")
    start_time = time.time()

    debiased_model = trainer.train_mlm(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_texts=val_texts,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=2,
        mlm_probability=0.15,
        max_length=128,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        save_total_limit=2,
        early_stopping=True,
        early_stopping_patience=3
    )

    training_time = time.time() - start_time

    # FIX: training_info path (saved under .../final/training_info.json)
    info_path = trainer.output_dir / "final" / "training_info.json"
    training_info = {}
    if info_path.exists():
        with open(info_path, 'r') as f:
            training_info = json.load(f)

    training_info['total_time_minutes'] = training_time / 60

    print(f"\nTotal training time: {training_time/60:.1f} minutes")

    return debiased_model, tokenizer, training_info


def run_reevaluation(
    evaluator: StereoSetEvaluator,
    manager: ModelManager,
    loader: DataLoader,
    stereoset_split: str = 'test',
    batch_size: int = 16,
    use_subset: bool = False,
    subset_size: int = 100
) -> dict:
    """
    Step 3: re-evaluate after debiasing
    Returns dict with debiased model metrics
    """
    print_step(3, "RE-EVALUATION (DEBIASED MODEL)")

    # load same stereoset data as baseline
    print("Loading StereoSet data...")
    stereoset_data = loader.load_stereoset(stereoset_split)

    if use_subset:
        stereoset_data['intrasentence'] = stereoset_data.get('intrasentence', [])[:subset_size]

    # load the debiased model
    print("\nLoading debiased model...")
    debiased_model, tokenizer = manager.load_model_from_path(
        str(manager.debiased_dir / "final")
    )

    # evaluate on stereoset again
    print("\nEvaluating debiased model on StereoSet...")
    start_time = time.time()

    debiased_results = evaluator.evaluate_stereoset(
        debiased_model, tokenizer, stereoset_data, batch_size=batch_size
    )

    eval_time = time.time() - start_time
    print(f"Evaluation time: {eval_time/60:.1f} minutes")

    # save results
    evaluator.save_results(
        debiased_results,
        "results/debiased_results.json",
        "roberta-base-debiased"
    )

    return debiased_results


def run_comparison(
    evaluator: StereoSetEvaluator,
    baseline_results: dict,
    debiased_results: dict
) -> dict:
    """
    Step 4: compare baseline vs debiased performance
    Returns dict with comparison metrics
    """
    print_step(4, "COMPARISON AND ANALYSIS")

    print("Computing metrics comparison...")
    comparison = evaluator.compare_results(baseline_results, debiased_results)

    # save comparison
    comp_path = Path("results/comparison_results.json")
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"Comparison saved: {comp_path}")

    # print summary
    overall = comparison['overall']
    print("\n" + "=" * 60)
    print("OVERALL COMPARISON SUMMARY")
    print("=" * 60)

    # show key metrics
    metrics = [
        ('Stereotype Score', 'ss'),
        ('Language Modeling Score', 'lms'),
        ('ICAT Score', 'icat')
    ]

    for metric_name, metric_key in metrics:
        base_val = overall['baseline'][metric_key]
        deb_val = overall['debiased'][metric_key]
        delta = overall['delta'][metric_key]

        print(f"\n{metric_name}:")
        print(f"  Baseline:  {base_val:.2f}")
        print(f"  Debiased:  {deb_val:.2f}")
        print(f"  Change:    {delta:+.2f}")

    # domain breakdown (FIX: use comparison['by_domain'])
    print("\n" + "-" * 60)
    print("DOMAIN-SPECIFIC RESULTS")
    print("-" * 60)

    for domain, dom_data in sorted(comparison.get('by_domain', {}).items()):
        base_ss = dom_data['baseline']['ss']
        deb_ss = dom_data['debiased']['ss']
        delta_ss = dom_data['delta']['ss']
        print(f"\n{domain.capitalize()}:")
        print(f"  Stereotype Score: {base_ss:.2f} → {deb_ss:.2f} ({delta_ss:+.2f})")

    return comparison


def run_visualization(
    baseline_results: dict,
    debiased_results: dict,
    training_info: dict,
    output_dir: str = 'results/plots'
):
    """
    Step 5: generate visualization charts
    Creates comparison plots and training progress charts
    """
    print_step(5, "VISUALIZATION")

    print("Generating visualizations...")

    # Usa la nuova API pulita - passa direttamente i dict dei risultati
    paths = visualize_all_results(
        baseline_results=baseline_results,
        mitigated_results=debiased_results,
        output_dir=output_dir
    )

    print(f"\nGenerated {len(paths)} visualizations:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
    
    print(f"\nAll visualizations saved to: {output_dir}/")


def print_final_summary(
    baseline_results: dict,
    debiased_results: dict,
    training_info: dict,
    total_time: float,
    visualizations_enabled: bool = True
):
    """Print comprehensive final summary of the experiment"""

    print("\n\n" + "=" * 70)
    print("FINAL PROJECT SUMMARY")
    print("=" * 70)

    # timing info
    print(f"\nTotal pipeline time: {total_time/60:.1f} minutes")
    if training_info:
        print(f"Training time: {training_info.get('total_time_minutes', 0):.1f} minutes")

    # key results
    base_overall = baseline_results['overall']
    deb_overall = debiased_results['overall']

    print("\n" + "-" * 70)
    print("KEY RESULTS")
    print("-" * 70)

    print("\nBaseline Model:")
    print(f"  Stereotype Score: {base_overall['ss']:.2f}")
    print(f"  LM Score: {base_overall['lms']:.2f}")
    print(f"  ICAT Score: {base_overall['icat']:.2f}")

    print("\nDebiased Model:")
    print(f"  Stereotype Score: {deb_overall['ss']:.2f}")
    print(f"  LM Score: {deb_overall['lms']:.2f}")
    print(f"  ICAT Score: {deb_overall['icat']:.2f}")

    # compute improvements
    ss_delta = deb_overall['ss'] - base_overall['ss']
    lms_delta = deb_overall['lms'] - base_overall['lms']
    icat_delta = deb_overall['icat'] - base_overall['icat']

    print("\nImprovements:")
    print(f"  Stereotype Score: {ss_delta:+.2f}")
    print(f"  LM Score: {lms_delta:+.2f}")
    print(f"  ICAT Score: {icat_delta:+.2f}")

    # files generated (più generico per riflettere l'API di visualizzazione attuale)
    print("\n" + "-" * 70)
    print("OUTPUT FILES")
    print("-" * 70)
    print("\nResults:")
    print("  results/baseline_results.json")
    print("  results/debiased_results.json")
    print("  results/comparison_results.json")
    print("  results/experiment_log.json")

    print("\nModels:")
    print("  models/baseline/original/")
    print("  models/debiased/final/")

    if visualizations_enabled:
        print("\nVisualizations:")
        print("  results/plots/*.png")

    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


def main():
    """Main pipeline entry point"""

    # argument parser
    parser = argparse.ArgumentParser(
        description="Mind the Gap - Bias Mitigation Pipeline"
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with subset of data'
    )
    parser.add_argument(
        '--stereoset-split',
        type=str,
        default='test',
        choices=['dev', 'test'],
        help='StereoSet split to use (default: test)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)'
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline evaluation (use existing results)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training (use existing debiased model)'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--only-visualize',
        action='store_true',
        help='Only generate visualizations from existing results'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='results/plots',
        help='Directory for saving plots (default: results/plots)'
    )

    args = parser.parse_args()

    # special mode: just regenerate visualizations
    if args.only_visualize:
        print_header("VISUALIZATION ONLY MODE")
        print("Loading existing results and generating visualizations...")

        try:
            # load saved results
            with open("results/baseline_results.json", 'r') as f:
                baseline_results = json.load(f)
            with open("results/debiased_results.json", 'r') as f:
                debiased_results = json.load(f)

            # optional training info
            training_info = {}
            try:
                training_info_path = Path("models/debiased/final/training_info.json")
                if training_info_path.exists():
                    with open(training_info_path, 'r') as f:
                        training_info = json.load(f)
            except:
                pass

            # generate visualizations
            run_visualization(
                baseline_results,
                debiased_results,
                training_info,
                output_dir=args.plot_dir
            )

            print("\nVisualizations generated successfully!")
            return 0

        except FileNotFoundError as e:
            print(f"\nError: Required result files not found")
            print(f"   {e}")
            print("\nPlease run the full pipeline first without --only-visualize")
            return 1
        except Exception as e:
            print(f"\nError generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # regular pipeline mode
    print_header("MIND THE GAP - BIAS MITIGATION PIPELINE")
    print("Project: Identifying and Mitigating Social Biases in RoBERTa")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.quick_test:
        print("\nQUICK TEST MODE: Using subset of data")

    if args.skip_visualization:
        print("Visualization disabled")

    # initialize all components
    print("\nInitializing components...")
    loader = DataLoader()
    manager = ModelManager()
    evaluator = StereoSetEvaluator()
    trainer = MLMTrainer()

    print("All components initialized")

    # start timer
    pipeline_start = time.time()

    # setup experiment tracking
    experiment_log = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'stereoset_split': args.stereoset_split,
            'batch_size': args.batch_size,
            'train_batch_size': args.train_batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'quick_test': args.quick_test,
            'visualization_enabled': not args.skip_visualization
        }
    }

    try:
        # Step 1: baseline evaluation
        if not args.skip_baseline:
            baseline_results = run_baseline_evaluation(
                evaluator, manager, loader,
                stereoset_split=args.stereoset_split,
                batch_size=args.batch_size,
                use_subset=args.quick_test,
                subset_size=100 if args.quick_test else 100  # default safe
            )
            experiment_log['baseline'] = baseline_results['overall']
        else:
            print_step(1, "BASELINE EVALUATION (SKIPPED)")
            print("Loading existing baseline results...")
            with open("results/baseline_results.json", 'r') as f:
                baseline_results = json.load(f)

        # Step 2: training
        if not args.skip_training:
            debiased_model, tokenizer, training_info = run_training(
                trainer, manager, loader,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
                batch_size=args.train_batch_size,
                use_subset=args.quick_test,
                subset_size=500 if args.quick_test else 500  # default safe
            )
            experiment_log['training'] = training_info
        else:
            print_step(2, "FINE-TUNING (SKIPPED)")
            print("Using existing debiased model...")
            training_info = {}

        # Step 3: re-evaluation
        debiased_results = run_reevaluation(
            evaluator, manager, loader,
            stereoset_split=args.stereoset_split,
            batch_size=args.batch_size,
            use_subset=args.quick_test,
            subset_size=100 if args.quick_test else 100  # default safe
        )
        experiment_log['debiased'] = debiased_results['overall']

        # Step 4: comparison
        comparison = run_comparison(
            evaluator, baseline_results, debiased_results
        )
        experiment_log['comparison'] = comparison['overall']

        # Step 5: visualization (if enabled)
        if not args.skip_visualization:
            run_visualization(
                baseline_results,
                debiased_results,
                training_info,
                output_dir=args.plot_dir
            )
        else:
            print_step(5, "VISUALIZATION (SKIPPED)")

        # calculate total time
        total_time = time.time() - pipeline_start
        experiment_log['total_time_minutes'] = total_time / 60

        # save experiment log
        save_experiment_log(experiment_log)

        # print final summary
        print_final_summary(
            baseline_results,
            debiased_results,
            training_info,
            total_time,
            visualizations_enabled=not args.skip_visualization
        )

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        print("Partial results may be saved in results/ directory")
        return 1

    except Exception as e:
        print(f"\n\nError during pipeline execution:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())