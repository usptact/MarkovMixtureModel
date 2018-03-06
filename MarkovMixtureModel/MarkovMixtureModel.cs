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
        private Range N;
        private Range T;
        private Range K;

        // model variables
        private VariableArray<VariableArray<int>, int[][]> States;

        // model parameters
        private Variable<Vector> ProbInit;
        private VariableArray<Vector> CPTTrans;

        // prior distributions
        private Variable<Dirichlet> ProbInitPrior;
        private VariableArray<Dirichlet> CPTTransPrior;

        // posteriors
        private Dirichlet[] CPTTransPosterior;
        private Dirichlet ProbInitPosterior;

        private InferenceEngine engine;


        public MarkovMixtureModel(int NumChains, int ChainLength, int NumStates)
        {
            // set ranges
            N = new Range(NumChains).Named("N");
            T = new Range(ChainLength).Named("T");
            K = new Range(NumStates).Named("K");

            // set init
            ProbInitPrior = Variable.New<Dirichlet>();
            ProbInit = Variable<Vector>.Random(ProbInitPrior).Named("ProbInit");
            ProbInit.SetValueRange(K);

            // set trans prob. table
            CPTTransPrior = Variable.Array<Dirichlet>(K);
            CPTTrans = Variable.Array<Vector>(K).Named("CPTTrans");
            CPTTrans[K] = Variable<Vector>.Random(CPTTransPrior[K]);
            CPTTrans.SetValueRange(K);

            // define primary model variables -- actual states
            States = Variable.Array(Variable.Array<int>(T), N).Named("States");

            using (Variable.ForEach(N))
            {
                using (var block = Variable.ForEach(T))
                {
                    var t = block.Index;

                    using (Variable.If(t == 0))
                        States[N][T].SetTo(Variable.Discrete(ProbInit));

                    var previousState = States[N][t - 1];

                    using (Variable.If(t > 0))
                    using (Variable.Switch(previousState))
                        States[N][T].SetTo(Variable.Discrete(CPTTrans[previousState]));
                }
            }

            engine = new InferenceEngine();
        }

        public void SetPriors(Dirichlet ProbInitPriorParamObs, Dirichlet[] CPTTransPriorObs)
        {
            ProbInitPrior.ObservedValue = ProbInitPriorParamObs;
            CPTTransPrior.ObservedValue = CPTTransPriorObs;
        }

        public void ObserveData(int[][] StatesData)
        {
            States.ObservedValue = StatesData;
        }

        public void InitializeStatesRandomly()
        {
            var StatesInit = Variable.Array(Variable.Array<Discrete>(T), N);
            StatesInit.ObservedValue = Util.ArrayInit(N.SizeAsInt,
                                                      n => Util.ArrayInit(T.SizeAsInt, t => Discrete.PointMass(Rand.Int(K.SizeAsInt), K.SizeAsInt)));
            States[N][T].InitialiseTo(StatesInit[N][T]);
        }

        public void InferPosteriors()
        {
            CPTTransPosterior = engine.Infer<Dirichlet[]>(CPTTrans);
            ProbInitPosterior = engine.Infer<Dirichlet>(ProbInit);

            Console.WriteLine("\nESTIMATED: ProbInit Posterior:");
            Console.WriteLine(ProbInitPosterior.GetMean());

            Console.WriteLine("\nESTIMATED: CPTTrans Posterior:");
            for (int i = 0; i < CPTTransPosterior.Length; i++)
                Console.WriteLine(CPTTransPosterior[i].GetMean());
        }
    }
}
