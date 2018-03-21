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
    }
}
