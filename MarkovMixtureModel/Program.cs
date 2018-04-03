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

            // get mode
            string mode = "";
            if (cmdInfo.GotOption("mode"))
                mode = cmdInfo.GetValue("mode");
            else
            {
                PrintHelpMessage();
                Environment.Exit(1);
            }

            if (string.Equals(mode, "train"))
                Train(cmdInfo);                 // train the model

            if (string.Equals(mode, "predict"))
                Predict(cmdInfo);               // predict cluster assignments
        }

        static void Train(CommandLineInfo cmdInfo)
        {
            // get path to data file
            string dataFilename = "";
            if (cmdInfo.GotOption("data"))
                dataFilename = cmdInfo.GetValue("data");
            else
            {
                PrintHelpMessage();
                Environment.Exit(1);
            }

            // get number of clusters
            int C = -1;
            if (cmdInfo.GotOption("clusters"))
                C = Int32.Parse(cmdInfo.GetValue("clusters"));
            else
            {
                PrintHelpMessage();
                Environment.Exit(1);
            }

            // get model filename
            string modelFilename = @"";
            if (cmdInfo.GotOption("model"))
                modelFilename = cmdInfo.GetValue("model");
            else
            {
                PrintHelpMessage();
                Environment.Exit(1);
            }

            Reader reader = new Reader(dataFilename);
            reader.Read();

            int[][] data = reader.GetData();
            int[] sizes = reader.GetSize();
            int K = reader.GetNumberOfStates();

            // get uniform priors
            MarkovMixtureModel.GetUniformPriors(C, K,
                                                out Dirichlet ClusterPriorObs,
                                                out Dirichlet[] ProbInitPriorObs,
                                                out Dirichlet[][] CPTTransPriorObs);

            // do model training
            MarkovMixtureModel model = new MarkovMixtureModel(C);

            model.SetPriors(ClusterPriorObs, ProbInitPriorObs, CPTTransPriorObs);
            model.ObserveData(data, sizes, K);
            model.InitializeStatesRandomly();
            model.InferPosteriors();

            // save the posteriors
            model.saveModel(modelFilename);
        }

        static void Predict(CommandLineInfo cmdInfo)
        {
            
        }

        static void PrintHelpMessage()
        {
            Console.WriteLine("Training:");
            Console.WriteLine("\tUsage: MarkovMixtureModel.exe -mode:train -data:<data> -clusters:<int> -model:<model>\n");
            Console.WriteLine("Prediction:");
            Console.WriteLine("\tUsage: MarkovMixtureModel.exe -mode:predict -model:<model> -data:<data> -predictions:<pred>\n");
        }
    }
}
