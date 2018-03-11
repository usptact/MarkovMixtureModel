using System;
using System.IO;
using System.Collections.Generic;

namespace MarkovMixtureModel
{
    public class Reader
    {
        public string FileName;
        public List<int[]> Data;

        public Reader(string FileName)
        {
            this.FileName = FileName;
        }

        public void Read()
        {
            string line;

            Data = new List<int[]>();

            StreamReader reader = new StreamReader(FileName);

            while((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrEmpty(line))
                    continue;
                else
                {
                    int[] states = ParseLine(line);
                    Data.Add(states);
                }
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

        // parses a single text line into array of integers
        private int[] ParseLine(string line)
        {
            string[] tokens = line.Split(' ');
            int numTokens = tokens.Length;
            int[] states = new int[numTokens];

            for (int i = 0; i < numTokens; i++)
                states[i] = Int32.Parse(tokens[i]);

            return states;
        }
    }
}
