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

            if (cmdInfo.Params.Length < 2)
            {
                PrintHelpMessage();
                Environment.Exit(1);
            }

            // get verb
            string verb = "";
            if (cmdInfo.GotOption("--train"))
                verb = cmdInfo.GetValue("--train");
            if (cmdInfo.GotOption("--predict"))
                verb = cmdInfo.GetValue("--predict");
            if (verb == "")
            {
                PrintHelpMessage();
                Environment.Exit(1);
            }

            //
            // Set parameters: number of clusters
            //

            int C = 4;

            //
            // Get data
            //

            string fileName = @"sequences.txt";

            Reader reader = new Reader(fileName);
            reader.Read();

            int[][] data = reader.GetData();
            int[] sizes = reader.GetSize();
            int K = reader.GetNumberOfStates();

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
        }

        static void PrintHelpMessage()
        {
            Console.WriteLine("Training:");
            Console.WriteLine("\tUsage: MarkovMixtureModel.exe --train --data <data> --model <model>\n");
            Console.WriteLine("Prediction:");
            Console.WriteLine("\tUsage: MarkovMixtureModel.exe --predict --model <model> --data <data> --predictions <pred>\n");
        }
    }
}
