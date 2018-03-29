using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace MarkovMixtureModel
{
    /// <summary>
    /// Command line parsing tool.
    /// 
    /// To Do:
    /// * add validation; specify valid options and their type, required options, parameters, etc. and validate.
    /// </summary>
    public class CommandLineInfo
    {
        readonly List<string> _plainParams = new List<string>();
        readonly Dictionary<string, bool> _boolVals = new Dictionary<string, bool>();
        readonly Dictionary<string, string> _stringVals = new Dictionary<string, string>();

        static bool debug;

        /// <summary>
        /// Set this to true if you want verbose debugging output.
        /// </summary>
        public static bool Debug
        {
            get
            {
                return debug;
            }
        }

        /// <summary>
        /// True if /? was passed.
        /// </summary>
        public bool HelpRequested
        {
            get
            {
                // Theoretically could allow a help syntax for particulars that works like
                // this: "Tools /?:helptopic"
                return GotOption("?");
            }
        }

        /// <summary>
        /// These all all the unadorned parameters that were passed on the comment line.
        /// For example, with the command line "one /two three /cheese" Params would be
        /// ["one", "three"].
        /// </summary>
        public string[] Params
        {
            get
            {
                return _plainParams.ToArray();
            }
        }

        public CommandLineInfo(IEnumerable<string> args)
        {
            var stringValPat = new Regex(@"^[/\-](.*?)\s*[\:\=]\s*(.*?)$", RegexOptions.Compiled);
            var boolValPat = new Regex(@"^[/\-](.*)", RegexOptions.Compiled);

            foreach (string arg in args)
            {
                if (stringValPat.IsMatch(arg))
                {
                    Match match = stringValPat.Match(arg);
                    string key = match.Groups[1].Value.ToLower();
                    string value = match.Groups[2].Value;
                    _stringVals[key] = value;
                }
                else
                    if (boolValPat.IsMatch(arg))
                    _boolVals.Add(boolValPat.Match(arg).Groups[1].Value.ToLowerInvariant(), true);
                else
                    _plainParams.Add(arg);
            }

            if (GotBoolOption("debug"))
                debug = true;
        }

        /// <summary>
        /// Tells you whether an option was specified, wether it was with or without a value.
        /// That is if the option was specified as a boolean option or a value option.
        /// 
        /// </summary>
        /// <param name="opt">The option name.</param>
        /// <returns>true if the option was specified.</returns>
        public bool GotOption(string opt)
        {
            return _boolVals.ContainsKey(opt.ToLowerInvariant()) || _stringVals.ContainsKey(opt.ToLowerInvariant());
        }

        /// <summary>
        /// Gets the value of a value option.
        /// </summary>
        /// <param name="opt">The name of the value option.</param>
        /// <returns>The value, if the option was specified, otherwise, null.</returns>
        public string GetValue(string opt)
        {
            string value = null;
            string key = opt.ToLowerInvariant();

            if (_stringVals.ContainsKey(key))
                value = _stringVals[key];

            return value;
        }

        /// <summary>
        /// Gets the value of a boolean option; you can specify multiple options or aliases.
        /// For example GotBoolOption("v", "verbose") returns true if either (or both) /v
        /// or /verbose are specified.
        /// </summary>
        /// <param name="optionNames">Name of the option.</param>
        /// <returns>true if any of the options were specified, false otherwise.</returns>
        public bool GotBoolOption(params string[] optionNames)
        {
            // Hmm... What's cleaner, specifying alias options as "opt1|opt2|opt3" or with params as ("opt1","opt2","opt3"...)
            return optionNames.Any(option => _boolVals.ContainsKey(option.ToLowerInvariant()));
        }

        /// <summary>
        /// Simple Print utility that just writes lines to the console.
        /// </summary>
        /// <param name="msg"></param>
        /// <param name="args"></param>
        private static void print(string msg, params object[] args)
        {
            Console.WriteLine(msg, args);
        }

        public void SelfTest()
        {
            if (_plainParams.Count > 0)
            {
                print("Unadorned parameter{0}:", _plainParams.Count == 1 ? "" : "s");
                foreach (string arg in _plainParams)
                    print("   {0}", arg);
            }
            else
                print("There were no unadorned parameters.");

            if (_boolVals.Count > 0)
            {
                print("Flag{0}:", _boolVals.Count == 1 ? "" : "s");
                foreach (string arg in _boolVals.Keys)
                    print("   {0}", arg);
            }
            else
                print("There were no flag (or boolean) options.");

            if (_stringVals.Count > 0)
            {
                print("String value{0}:", _stringVals.Count == 1 ? "" : "s");
                foreach (string arg in _stringVals.Keys)
                    print("   {0} = \"{1}\"", arg, _stringVals[arg]);
            }
            else
                print("There were no string values specified.");

            // a few random tests:
            if (GotBoolOption("?"))
                print("Help was requested.");
            if (GotOption("blah"))
                print("blah option was specified.");
            if (GotOption("BLAH"))
                print("BLAH option was specified.");

            if (GotOption("cheese"))
                print("cheese is \"{0}\"", GetValue("cheese"));
            else
                print("cheese was not specified.");

            print(GotBoolOption("cheese") ? "cheese is a bool option." : "cheese is not a bool option.");
        }
    }
}
