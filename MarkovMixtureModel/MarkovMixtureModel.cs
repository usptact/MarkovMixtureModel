using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Distributions;

namespace MarkovMixtureModel
{
    public class MarkovMixtureModel
    {
        // state data
        private int[][] StatesData;

        // ranges
        private Range N;
        private Range T;
        private Range K;

        // model variables
        private VariableArray<int> ZeroStates;
        private VariableArray<VariableArray<int>, int[][]> States;

        // model parameters
        private Variable<Vector> ProbInit;
        private VariableArray<Vector> CPTTrans;

        // prior distributions
        private Variable<Dirichlet> ProbInitPrior;
        private VariableArray<Dirichlet> CPTTransPrior;

        private InferenceEngine engine;


        public MarkovMixtureModel(int NumChains, int ChainLength, int NumStates)
        {
            // set ranges
            N = new Range(NumChains);
            T = new Range(ChainLength);
            K = new Range(NumStates);

            // set init
            ProbInitPrior = Variable.New<Dirichlet>();
            ProbInit = Variable<Vector>.Random(ProbInitPrior);
            ProbInit.SetValueRange(K);

            // set trans prob. table
            CPTTransPrior = Variable.Array<Dirichlet>(K);
            CPTTrans = Variable.Array<Vector>(K);
            CPTTrans[K] = Variable<Vector>.Random(CPTTransPrior[K]);
            CPTTrans.SetValueRange(K);

            // define primary model variables -- zero state
            ZeroStates = Variable.Array<int>(N);
            ZeroStates[N] = Variable.Discrete(ProbInit).ForEach(N);

            // define primary model variables -- actual states
            States = Variable.Array(Variable.Array<int>(T), N);

            using (Variable.ForEach(N))
            {
                using (var block = Variable.ForEach(T))
                {
                    var t = block.Index;
                    var previousState = States[N][t - 1];

                    using (Variable.If(t == 0))
                    {
                        using (Variable.Switch(ZeroStates[N]))
                            States[N][T] = Variable.Discrete(CPTTrans[ZeroStates[N]]);
                    }

                    using (Variable.If(t > 0))
                    {
                        using (Variable.Switch(previousState))
                            States[N][T] = Variable.Discrete(CPTTrans[previousState]);
                    }
                }
            }

            engine = new InferenceEngine();
        }

        public void SetPriors(Dirichlet ProbInitPriorParamObs, Dirichlet[] CPTTransPriorObs)
        {
            ProbInitPrior.ObservedValue = ProbInitPriorParamObs;
            CPTTransPrior.ObservedValue = CPTTransPriorObs;
        }
    }
}
