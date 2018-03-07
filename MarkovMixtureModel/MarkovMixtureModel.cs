using System;
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
        private Range C;
        private Range N;
        private Range T;
        private Range K;

        // model variables
        private VariableArray<VariableArray<int>, int[][]> States;

        // model parameters
        private Variable<Vector> ProbCluster;
        private VariableArray<Vector> ProbInit;
        private VariableArray<VariableArray<Vector>, Vector[][]> CPTTrans;

        // aux model variables
        private VariableArray<int> Z;

        // prior distributions
        private Variable<Dirichlet> ProbClusterPrior;
        private VariableArray<Dirichlet> ProbInitPrior;
        private VariableArray<VariableArray<Dirichlet>, Dirichlet[][]> CPTTransPrior;

        // posteriors
        private Dirichlet ProbClusterPosterior;
        private Dirichlet[][] CPTTransPosterior;
        private Dirichlet[] ProbInitPosterior;

        private InferenceEngine engine;


        public MarkovMixtureModel(int NumClusters, int NumChains, int ChainLength, int NumStates)
        {
            // set ranges
            C = new Range(NumClusters).Named("C");
            N = new Range(NumChains).Named("N");
            T = new Range(ChainLength).Named("T");
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
            for (int i = 0; i < NumClusters; i++)
                CPTTrans[i].SetValueRange(K);

            // define primary model variables -- actual states
            States = Variable.Array(Variable.Array<int>(T), N).Named("States");
            for (int i = 0; i < NumChains; i++)
                States[i].SetValueRange(K);

            // define aux model variable
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

        public void SetPriors(Dirichlet ClusterPriorParamObs, Dirichlet[] ProbInitPriorParamObs, Dirichlet[][] CPTTransPriorObs)
        {
            ProbClusterPrior.ObservedValue = ClusterPriorParamObs;
            ProbInitPrior.ObservedValue = ProbInitPriorParamObs;
            CPTTransPrior.ObservedValue = CPTTransPriorObs;
        }

        public void ObserveData(int[][] StatesData)
        {
            States.ObservedValue = StatesData;
        }

        public void InitializeStatesRandomly()
        {
            // random cluster assignments
            Discrete[] Zinit = new Discrete[N.SizeAsInt];
            for (int i = 0; i < Zinit.Length; i++)
                Zinit[i] = Discrete.PointMass(Rand.Int(C.SizeAsInt), C.SizeAsInt);
            Z.InitialiseTo(Distribution<int>.Array(Zinit));

            /*
            var StatesInit = Variable.Array(Variable.Array<Discrete>(T), N);
            StatesInit.ObservedValue = Util.ArrayInit(N.SizeAsInt,
                                                      n => Util.ArrayInit(T.SizeAsInt, t => Discrete.PointMass(Rand.Int(K.SizeAsInt), K.SizeAsInt)));
            States[N][T].InitialiseTo(StatesInit[N][T]);
            */
        }

        public void InferPosteriors()
        {
            ProbClusterPosterior = engine.Infer<Dirichlet>(ProbCluster);
            CPTTransPosterior = engine.Infer<Dirichlet[][]>(CPTTrans);
            ProbInitPosterior = engine.Infer<Dirichlet[]>(ProbInit);

            Console.WriteLine("\n === PARAMETER ESTIMATES ===");
            Console.WriteLine("ProbClusterPosterior: {0}", ProbClusterPosterior.GetMean());

            for (int c = 0; c < C.SizeAsInt; c++)
            {
                Console.WriteLine("\t=== CLUSTER #{0} ===", c);

                Console.WriteLine("\tProbInit Posterior: {0}", ProbInitPosterior[c].GetMean());

                Console.WriteLine("\n\tCPTTrans Posterior:");
                for (int i = 0; i < CPTTransPosterior[c].Length; i++)
                    Console.WriteLine("\t{0}", CPTTransPosterior[c][i].GetMean());
            }
        }
    }
}
