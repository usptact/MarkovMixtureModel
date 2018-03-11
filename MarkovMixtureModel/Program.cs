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
            //TestMarkovMixtureModel();

            string fileName = @"sample.txt";

            Reader reader = new Reader(fileName);
            reader.Read();

            int[][] data = reader.GetData();
            int[] sizes = reader.GetSize();
        }

        public static void TestMarkovMixtureModel()
        {
            Rand.Restart(2019);

            int C = 4;          // number of clusters
            int N = 100;       // total number of sequences
            int T = 100;        // sequence length
            int K = 3;          // total number of states

            //
            // hyperparameters
            //

            Dirichlet ClusterPriorObs = Dirichlet.Uniform(C);
            Dirichlet[] ProbInitPriorObs = Enumerable.Repeat(Dirichlet.Uniform(K), C).ToArray();
            Dirichlet[][] CPTTransPriorObs = Enumerable.Repeat(Enumerable.Repeat(Dirichlet.Uniform(K), K).ToArray(), C).ToArray();

            //
            // sample model parameters
            //

            //double[] clusterProbs = ClusterPriorObs.Sample().ToArray();
            double[] clusterProbs = new double[]{ 0.05, 0.15, 0.4, 0.4 };

            Console.WriteLine("=== TRUE PARAMETERS ===");
            Console.WriteLine("ClusterProbs:");
            Console.WriteLine("\t[{0:0.###}]", string.Join(" ", clusterProbs));

            double[][] init = new double[C][];
            double[][][] trans = new double[C][][];
            int[][] states = new int[N][];

            int counter = 0;
            for (int c = 0; c < C; c++)
            {
                Console.WriteLine("=== CLUSTER #{0} ===", c);

                // number of points to generate in this cluster
                int nc = (int)Math.Round(N * clusterProbs[c]);

                // current cluster init state probabilities
                init[c] = ProbInitPriorObs[c].Sample().ToArray();

                Console.WriteLine("\n\tInit:");
                Console.WriteLine("\t[{0}]", string.Join(" ", init[c]));

                // current cluster transition probability matrix
                trans[c] = new double[K][];
                for (int i = 0; i < K; i++)
                    trans[c][i] = CPTTransPriorObs[c][i].Sample().ToArray();

                Console.WriteLine("\n\tCPTTrans:");
                for (int i = 0; i < trans[c].Length; i++)
                    Console.WriteLine("\t[{0}]", string.Join(" ", trans[c][i]));

                int[][] cstates = GenerateData(init[c], trans[c], T, nc);
                for (int i = 0; i < nc; i++)
                {
                    states[counter] = new int[T];
                    for (int j = 0; j < T; j++)
                    {
                        states[counter][j] = cstates[i][j];
                    }
                    counter++;
                }
            }

            Console.WriteLine();

            MarkovMixtureModel model = new MarkovMixtureModel(C, N, T, K);

            model.SetPriors(ClusterPriorObs, ProbInitPriorObs, CPTTransPriorObs);
            model.ObserveData(states);
            model.InitializeStatesRandomly();
            model.InferPosteriors();

            Discrete[] ClusterAssignments = model.GetClusterAssignments();

            Console.WriteLine("\n=== CLUSTER ASSIGNMENTS ===");
            for (int i = 0; i < N; i++)
                Console.WriteLine(ClusterAssignments[i].GetProbs());
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
