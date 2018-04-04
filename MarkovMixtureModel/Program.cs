using System;
using System.Linq;
using System.IO;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MarkovMixtureModel
{
    class Program
    {
        static void Main(string[] args)
        {
            var cmdInfo = new CommandLineInfo(args);
            GetArguments(cmdInfo, out string dataFilename, out int numClusters, out string predFilename);

            if (dataFilename == "" || numClusters == -1 || predFilename == "")
            {
                PrintHelpMessage();
                Environment.Exit(1);
            }

            Reader reader = new Reader(dataFilename);
            reader.Read();

            int[][] data = reader.GetData();
            int[] sizes = reader.GetSize();
            int numStates = reader.GetNumberOfStates();

            // get uniform priors
            MarkovMixtureModel.GetUniformPriors(numClusters, numStates,
                                                out Dirichlet ClusterPriorObs,
                                                out Dirichlet[] ProbInitPriorObs,
                                                out Dirichlet[][] CPTTransPriorObs);

            // do model training
            MarkovMixtureModel model = new MarkovMixtureModel(numClusters);

            model.SetPriors(ClusterPriorObs, ProbInitPriorObs, CPTTransPriorObs);
            model.ObserveData(data, sizes, numStates);
            model.InitializeStatesRandomly();
            model.InferPosteriors();

            Console.WriteLine("\n=== Cluster Assignments ===");
            Console.WriteLine("Writing to: {0}", predFilename);
            Vector[] assignments = model.GetClusterAssignments();
            WriteClusterAssignments(predFilename, assignments);
        }

        public static void GetArguments(CommandLineInfo cmdInfo,
                                out string dataFilename,
                                out int numClusters,
                                out string predFilename)
        {
            dataFilename = @"";
            if (cmdInfo.GotOption("data"))
                dataFilename = cmdInfo.GetValue("data");

            numClusters = -1;
            if (cmdInfo.GotOption("clusters"))
                numClusters = Int32.Parse(cmdInfo.GetValue("clusters"));

            predFilename = @"";
            if (cmdInfo.GotOption("predictions"))
                predFilename = cmdInfo.GetValue("predictions");
        }

        public static void WriteClusterAssignments(string predFilename, Vector[] assignments)
        {
            StreamWriter writer = new StreamWriter(predFilename);
            for (int i = 0; i < assignments.Length; i++)
                writer.WriteLine(assignments[i]);
            writer.Close();
        }

        public static void PrintHelpMessage()
        {
            Console.WriteLine("Training:");
            Console.WriteLine("\tUsage: MarkovMixtureModel.exe -data:<path> -clusters:<int> -predictions:<path>\n");
        }
    }
}
