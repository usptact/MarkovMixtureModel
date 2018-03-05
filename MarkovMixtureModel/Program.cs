using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MarkovMixtureModel
{
    class Program
    {
        static void Main(string[] args)
        {
            TestMarkovMixtureModel();
        }

        public static void TestMarkovMixtureModel()
        {
            Rand.Restart(2018);

            int N = 50;
            int T = 10;
            int K = 3;

            // hyperparameters
            Dirichlet ProbInitPriorObs = Dirichlet.Uniform(K);
            Dirichlet[] CPTTransPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray();

            // sample model parameters
            double[] init = ProbInitPriorObs.Sample().ToArray();
            double[][] trans = new double[K][];
            for (int i = 0; i < K; i++)
                trans[i] = CPTTransPriorObs[i].Sample().ToArray();

            // generate some data
            int[][] data = GenerateData(init, trans, T, N);

            MarkovMixtureModel model = new MarkovMixtureModel(N, T, K);

            model.SetPriors(ProbInitPriorObs, CPTTransPriorObs);
            model.ObserveData(data);
            model.InitializeStatesRandomly();
            model.InferPosteriors();

            Console.ReadKey();
        }

        public static int[][] GenerateData(double[] init, double[][] trans, int T, int N)
        {
            int K = init.Length;

            // initial and transition distributions
            Discrete initDist = new Discrete(init);
            Discrete[] transDist = new Discrete[K];
            for (int i = 0; i < K; i++)
                transDist[i] = new Discrete(trans[i]);

            // sampling data
            int[][] actualStates = new int[N][];
            for (int n = 0; n < N; n++)
            {
                actualStates[n] = new int[T];
                actualStates[n][0] = initDist.Sample();
                for (int i = 1; i < T; i++)
                    actualStates[n][i] = transDist[actualStates[n][i - 1]].Sample();
            }
            return actualStates;
        }
    }
}
