using System;
using System.Linq;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Utils;


namespace MarkovMixtureModel
{
    public class MarkovMixtureModel
    {
        // ranges
        Range C;
        Range N;
        Range T;
        Range K;

        // model variables
        Variable<int> NumClusters;
        Variable<int> NumSequences;
        Variable<int> NumStates;
        VariableArray<int> SequenceSizes;
        VariableArray<VariableArray<int>, int[][]> States;

        // model parameters
        Variable<Vector> ProbCluster;
        VariableArray<Vector> ProbInit;
        VariableArray<VariableArray<Vector>, Vector[][]> CPTTrans;

        // aux model variables
        VariableArray<int> Z;

        // prior distributions
        Variable<Dirichlet> ProbClusterPrior;
        VariableArray<Dirichlet> ProbInitPrior;
        VariableArray<VariableArray<Dirichlet>, Dirichlet[][]> CPTTransPrior;

        // posteriors
        Dirichlet ProbClusterPosterior;
        Dirichlet[][] CPTTransPosterior;
        Dirichlet[] ProbInitPosterior;

        InferenceEngine engine;


        public MarkovMixtureModel(int NumberOfClusters)
        {
            NumClusters = Variable.Observed(NumberOfClusters);
            NumSequences = Variable.New<int>();
            NumStates = Variable.New<int>();

            // set ranges
            C = new Range(NumClusters).Named("C");
            K = new Range(NumStates).Named("K");

            // set cluster
            ProbClusterPrior = Variable.New<Dirichlet>();
            ProbCluster = Variable<Vector>.Random(ProbClusterPrior).Named("ProbCluster");
            ProbCluster.SetValueRange(C);

            // set init
            ProbInitPrior = Variable.Array<Dirichlet>(C);
            ProbInit = Variable.Array<Vector>(C).Named("ProbInit");
            ProbInit[C] = Variable<Vector>.Random(ProbInitPrior[C]).Named("ProbInit");
            ProbInit.SetValueRange(C);

            // set trans prob. table
            CPTTransPrior = Variable.Array(Variable.Array<Dirichlet>(K), C);
            CPTTrans = Variable.Array(Variable.Array<Vector>(K), C).Named("CPTTrans");
            CPTTrans[C][K] = Variable<Vector>.Random(CPTTransPrior[C][K]);
            CPTTrans.SetValueRange(K);

            // define jagged array sizes and ranges

            N = new Range(NumSequences);
            SequenceSizes = Variable.Array<int>(N);
            T = new Range(SequenceSizes[N]);

            // define primary model variables -- actual states
            States = Variable.Array(Variable.Array<int>(T), N).Named("States");
            States.SetValueRange(K);

            // define cluster assignment array
            Z = Variable.Array<int>(N);

            using (Variable.ForEach(N))
            {
                Z[N] = Variable.Discrete(ProbCluster);

                using (var block = Variable.ForEach(T))
                {
                    var t = block.Index;

                    using (Variable.If(t == 0))
                    using (Variable.Switch(Z[N]))
                        States[N][T].SetTo(Variable.Discrete(ProbInit[Z[N]]));

                    var previousState = States[N][t - 1];

                    using (Variable.If(t > 0))
                    using (Variable.Switch(previousState))
                    using (Variable.Switch(Z[N]))
                        States[N][T].SetTo(Variable.Discrete(CPTTrans[Z[N]][previousState]));
                }
            }

            engine = new InferenceEngine();
            engine.Compiler.UseParallelForLoops = true;
        }

        // return uniform priors
        public static void GetUniformPriors(int NumberOfClusters, int NumberOfStates,
                                            out Dirichlet ClusterPriorObs,
                                            out Dirichlet[] ProbInitPriorObs,
                                            out Dirichlet[][] CPTTransPriorObs)
        {
            ClusterPriorObs = Dirichlet.Uniform(NumberOfClusters);
            ProbInitPriorObs = Enumerable.Repeat(Dirichlet.Uniform(NumberOfStates), NumberOfClusters).ToArray();
            CPTTransPriorObs = Enumerable.Repeat(Enumerable.Repeat(Dirichlet.Uniform(NumberOfStates), NumberOfStates).ToArray(), NumberOfClusters).ToArray();
        }

        // set model priors
        public void SetPriors(Dirichlet ClusterPriorParamObs, Dirichlet[] ProbInitPriorParamObs, Dirichlet[][] CPTTransPriorObs)
        {
            ProbClusterPrior.ObservedValue = ClusterPriorParamObs;
            ProbInitPrior.ObservedValue = ProbInitPriorParamObs;
            CPTTransPrior.ObservedValue = CPTTransPriorObs;
        }

        // set data
        public void ObserveData(int[][] StatesData, int[] StatesSizes, int NumberOfStates)
        {
            NumStates.ObservedValue = NumberOfStates;
            NumSequences.ObservedValue = StatesData.Length;
            SequenceSizes.ObservedValue = StatesSizes;
            States.ObservedValue = StatesData;
        }

        // initialize cluster assignments at random
        public void InitializeStatesRandomly()
        {
            // random cluster assignments
            Discrete[] Zinit = new Discrete[NumSequences.ObservedValue];
            for (int i = 0; i < Zinit.Length; i++)
                Zinit[i] = Discrete.PointMass(Rand.Int(NumClusters.ObservedValue), NumClusters.ObservedValue);
            Z.InitialiseTo(Distribution<int>.Array(Zinit));
        }

        // infer unobserved model parameters
        public void InferPosteriors()
        {
            ProbClusterPosterior = engine.Infer<Dirichlet>(ProbCluster);
            CPTTransPosterior = engine.Infer<Dirichlet[][]>(CPTTrans);
            ProbInitPosterior = engine.Infer<Dirichlet[]>(ProbInit);

            Console.WriteLine("=== PARAMETER ESTIMATES ===");
            Console.WriteLine("Cluster Probabilities: {0}", ProbClusterPosterior.GetMean());

            for (int c = 0; c < NumClusters.ObservedValue; c++)
            {
                Console.WriteLine("\n=== Cluster #{0} ===", c);
                Console.WriteLine("Init: {0}", ProbInitPosterior[c].GetMean());
                for (int i = 0; i < CPTTransPosterior[c].Length; i++)
                    Console.WriteLine(CPTTransPosterior[c][i].GetMean());
            }
        }

        // infer cluster assigments
        public Discrete[] GetClusterAssignments()
        {
            Discrete[] ClusterAssignments = engine.Infer<Discrete[]>(Z);
            return ClusterAssignments;
        }

        // write posteriors to a file
        public void saveModel(string modelFilename)
        {
            BinaryFormatter serializer = new BinaryFormatter();

            using (FileStream stream = new FileStream(modelFilename, FileMode.Create))
            {
                serializer.Serialize(stream, ProbClusterPosterior);
                serializer.Serialize(stream, CPTTransPosterior);
                serializer.Serialize(stream, ProbInitPosterior);
            }
        }

        // read posteriors file
        public void readModel(string modelFilename,
                              out Dirichlet inProbClusterPosterior,
                              out Dirichlet[][] inCPTTransPosterior,
                              out Dirichlet[] inProbInitPosterior)
        {
            BinaryFormatter serializer = new BinaryFormatter();

            using (FileStream stream = new FileStream(modelFilename, FileMode.Create))
            {
                inProbClusterPosterior = (Dirichlet)serializer.Deserialize(stream);
                inCPTTransPosterior = (Dirichlet[][])serializer.Deserialize(stream);
                inProbInitPosterior = (Dirichlet[])serializer.Deserialize(stream);
            }
        }
    }
}
