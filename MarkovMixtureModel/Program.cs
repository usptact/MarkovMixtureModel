using System;
using System.Linq;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MarkovMixtureModel
{
    class Program
    {
        static void Main(string[] args)
        {
            int NumPoints = 1000;
            int NumClusters = 3;
            int NumStates = 2;

            GenerateModelParameters(NumClusters, NumStates,
                                    out Discrete clusterProbs,
                                    out Discrete[] initProbs,
                                    out Discrete[][] transProbs);

            int[][] data = GenerateData(NumPoints,
                                        clusterProbs,
                                        initProbs,
                                        transProbs);

            /*
            //
            // Get data
            //

            string fileName = @"sample.txt";

            Reader reader = new Reader(fileName);
            reader.Read();

            int[][] data = reader.GetData();
            int[] sizes = reader.GetSize();
            int K = reader.GetNumberOfStates();

            //
            // Set parameters: number of clusters
            //

            int C = 4;

            //
            // Set priors
            //

            MarkovMixtureModel.GetUniformPriors(C, K,
                                                out Dirichlet ClusterPriorObs,
                                                out Dirichlet[] ProbInitPriorObs,
                                                out Dirichlet[][] CPTTransPriorObs);

            //
            // Model training
            //

            MarkovMixtureModel model = new MarkovMixtureModel(C);

            model.SetPriors(ClusterPriorObs, ProbInitPriorObs, CPTTransPriorObs);
            model.ObserveData(data, sizes, K);
            model.InitializeStatesRandomly();
            model.InferPosteriors();
            */

            Console.WriteLine();
        }

        // generate model parameters
        public static void GenerateModelParameters(int NumClusters, int NumStates,
                                                   out Discrete clusterProbs,
                                                   out Discrete[] initProbs,
                                                   out Discrete[][] transProbs)
        {
            // cluster proportions
            Dirichlet clusterDist = Dirichlet.Uniform(NumClusters);
            double[] clusterProbsParams = clusterDist.Sample().ToArray();
            clusterProbs = new Discrete(clusterProbsParams);

            // sample cluster-specific parameters
            initProbs = new Discrete[NumClusters];
            transProbs = new Discrete[NumClusters][];
            for (int c = 0; c < NumClusters; c++)
            {
                Dirichlet cInit = Dirichlet.Uniform(NumStates);
                double[] initProbsParam = cInit.Sample().ToArray();
                initProbs[c] = new Discrete(initProbsParam);

                Dirichlet transDist = Dirichlet.Uniform(NumStates);
                transProbs[c] = new Discrete[NumStates];
                for (int k = 0; k < NumStates; k++)
                {
                    double[] transProbsParam = transDist.Sample().ToArray();
                    transProbs[c][k] = new Discrete(transProbsParam);
                }
            }
        }

        // generate data given model parameters
        public static int[][] GenerateData(int NumPoints,
                                           Discrete clusterProbs,
                                           Discrete[] initProbs,
                                           Discrete[][] transProbs)
        {
            int[][] data = new int[NumPoints][];

            int NumClusters = initProbs.Length;

            Poisson seqLengthDist = new Poisson(5);

            for (int i = 0; i < NumPoints; i++)
            {
                int cluster = clusterProbs.Sample();

                int seqLength = seqLengthDist.Sample() + 3;

                int[] seq = new int[seqLength];
                for (int t = 0; t < seqLength; t++)
                {
                    if (t == 0)
                        seq[0] = initProbs[cluster].Sample();
                    else
                        seq[t] = transProbs[cluster][seq[t - 1]].Sample();
                }
                data[i] = seq;
            }

            return data;
        }
    }
}
