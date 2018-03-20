using System.IO;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace SyntheticData
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            int NumPoints = 10000;
            int NumClusters = 4;
            int NumStates = 3;

            System.Console.WriteLine("===========================================================");
            System.Console.WriteLine("Generating {0} data points in {1} clusters and {2} states.", 
                                     NumPoints, NumClusters, NumStates);
            System.Console.WriteLine("===========================================================");

            GenerateModelParameters(NumClusters, NumStates,
                                    out Discrete clusterProbs,
                                    out Discrete[] initProbs,
                                    out Discrete[][] transProbs);

            PrintModelParameters(clusterProbs, initProbs, transProbs);

            int[][] data = GenerateData(NumPoints,
                                        clusterProbs,
                                        initProbs,
                                        transProbs);

            StreamWriter file = new StreamWriter("sample.txt");
            for (int i = 0; i < NumPoints; i++)
            {
                string line = string.Join(" ", data[i]);
                file.WriteLine(line);
            }
            file.Close();
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

        // prints model parameters
        public static void PrintModelParameters(Discrete clusterProbs,
                                                Discrete[] initProbs,
                                                Discrete[][] transProbs)
        {
            System.Console.WriteLine("Cluster probabilities: {0}", clusterProbs.GetProbs());
            int NumClusters = initProbs.Length;
            for (int c = 0; c < NumClusters; c++)
            {
                System.Console.WriteLine("\n=== Cluster #{0} ===", c);
                System.Console.WriteLine("Init: {0}", initProbs[c].GetProbs());
                for (int k = 0; k < transProbs[c][0].Dimension; k++)
                {
                    System.Console.WriteLine(transProbs[c][k].GetProbs());
                }
            }
        }
    }
}
