using System;
using System.IO;
using System.Collections.Generic;

namespace MarkovMixtureModel
{
    public class Reader
    {
        string FileName;
        List<int[]> Data;
        int NumberOfStates;

        public Reader(string fName)
        {
            FileName = fName;
            Data = new List<int[]>();
            NumberOfStates = -1;
        }

        public void Read()
        {
            string line;
            StreamReader reader = new StreamReader(FileName);
            while((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrEmpty(line))
                    continue;
                int[] states = ParseLine(line);
                Data.Add(states);
            }
            reader.Close();
        }

        // get data as array of arrays
        public int[][] GetData()
        {
            int numLines = Data.Count;
            int[][] data = new int[numLines][];
            for (int i = 0; i < numLines; i++)
            {
                int[] seq = Data[i];
                int seqLength = seq.Length;
                data[i] = new int[seqLength];

                for (int j = 0; j < seqLength; j++)
                    data[i][j] = seq[j];
            }
            return data;
        }

        // get sequence sizes
        public int[] GetSize()
        {
            int numLines = Data.Count;
            int[] sizes = new int[numLines];
            for (int i = 0; i < numLines; i++)
                sizes[i] = Data[i].Length;
            return sizes;
        }

        public int GetNumberOfStates()
        {
            return NumberOfStates;
        }

        // parses a single text line into array of integers
        int[] ParseLine(string line)
        {
            string[] tokens = line.Split(' ');
            int numTokens = tokens.Length;
            int[] states = new int[numTokens];
            for (int i = 0; i < numTokens; i++) {
                states[i] = Int32.Parse(tokens[i]);
                NumberOfStates = Math.Max(NumberOfStates, states[i]);
            }
            return states;
        }
    }
}
