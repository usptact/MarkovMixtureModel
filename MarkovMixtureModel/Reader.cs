using System;
using System.IO;

namespace MarkovMixtureModel
{
    public class Reader
    {
        public string FileName;
        public int[][] Data;
        public int[] Sizes;

        public Reader(string FileName)
        {
            this.FileName = FileName;
        }

        public void Read()
        {
            string line;

            StreamReader reader = new StreamReader(FileName);

            while((line = reader.ReadLine()) != null)
            {
                int[] states = ParseLine(line);
            }

            reader.Close();
        }

        int[] ParseLine(string line)
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
