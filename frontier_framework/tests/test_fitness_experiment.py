import unittest

from frontier_framework import Civilization, FitnessExperiment


class FitnessExperimentTests(unittest.TestCase):
    def test_records_real_pre_and_post_tick_fitness(self):
        experiment = FitnessExperiment()
        civilization = Civilization(
            n_nodes=1,
            node_population_size=4,
            exchange_rate=0.0,
            fitness_experiment=experiment,
        )
        pre_fitness = civilization._average_fitness()

        result = civilization.tick()

        self.assertEqual(len(experiment.observations), 1)
        observation = experiment.observations[0]
        self.assertEqual(observation.run_id, civilization.id)
        self.assertEqual(observation.cycle, 0)
        self.assertEqual(observation.features["avg_fitness"], pre_fitness)
        self.assertAlmostEqual(observation.target_avg_fitness, civilization._average_fitness())
        self.assertAlmostEqual(observation.target_avg_fitness, result["avg_fitness"], places=4)

    def test_baseline_evaluates_only_complete_held_out_runs(self):
        experiment = FitnessExperiment()
        train = Civilization(
            n_nodes=1,
            node_population_size=4,
            exchange_rate=0.0,
            fitness_experiment=experiment,
        )
        held_out = Civilization(
            n_nodes=1,
            node_population_size=4,
            exchange_rate=0.0,
            fitness_experiment=experiment,
        )
        train.tick()
        train.tick()
        held_out.tick()
        held_out.tick()

        result = experiment.evaluate_persistence_baseline([held_out.id])

        self.assertEqual(result["target"], "next_cycle_average_fitness")
        self.assertEqual(result["baseline"], "predict_current_average_fitness")
        self.assertEqual(result["metric"], "mae")
        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["non_held_out_run_count"], 1)
        self.assertEqual(result["held_out_run_count"], 1)
        self.assertEqual(result["held_out_run_ids"], [held_out.id])
        self.assertGreaterEqual(result["mae"], 0.0)

    def test_rejects_evaluation_without_a_separate_run(self):
        experiment = FitnessExperiment()
        civilization = Civilization(
            n_nodes=1,
            node_population_size=2,
            exchange_rate=0.0,
            fitness_experiment=experiment,
        )
        civilization.tick()

        with self.assertRaisesRegex(ValueError, "non-held-out run"):
            experiment.evaluate_persistence_baseline([civilization.id])


if __name__ == "__main__":
    unittest.main()
