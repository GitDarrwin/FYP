import { useState } from 'react';
import { AlertCircle, FileText, Shield, Upload, CheckCircle, XCircle } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";

interface ScanResult {
  confidence: number;
  status: 'clean' | 'malicious' | 'unknown';
  details: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<ScanResult | null>(null);
  const { toast } = useToast();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
    }
  };

  const simulateScan = async () => {
    if (!file) return;

    setScanning(true);
    setProgress(0);

    try {
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 500));
        setProgress(i);
      }

      const mockResult: ScanResult = {
        confidence: Math.random() * 100,
        status: Math.random() > 0.5 ? 'clean' : 'malicious',
        details: 'Detailed analysis of file behavior and characteristics'
      };

      setResult(mockResult);
      toast({
        title: "Scan Complete",
        description: `Analysis finished for ${file.name}`,
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to analyze file. Please try again.",
      });
    } finally {
      setScanning(false);
      setProgress(100);
    }
  };

  return (
    <div className="min-h-screen bg-background p-8 dark">
      <div className="max-w-2xl mx-auto space-y-8">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-2"
        >
          <Shield className="w-16 h-16 mx-auto text-primary animate-pulse" />
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-cyan-500">
            Malware Detection
          </h1>
          <p className="text-muted-foreground">
            Upload a file to analyze for potential security threats
          </p>
        </motion.div>

        <Card className="p-8 border-2 border-muted">
          <div className="space-y-6">
            <AnimatePresence>
              {!file ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center justify-center w-full"
                >
                  <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary hover:bg-secondary/10 transition-all duration-300">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-10 h-10 mb-3 text-primary" />
                      <p className="mb-2 text-sm text-muted-foreground">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-muted-foreground/70">Any file type accepted</p>
                    </div>
                    <input
                      type="file"
                      className="hidden"
                      onChange={handleFileChange}
                      disabled={scanning}
                    />
                  </label>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 bg-secondary/20 rounded-lg"
                >
                  <div className="flex items-center space-x-4">
                    <FileText className="w-8 h-8 text-primary" />
                    <div className="flex-1">
                      <h3 className="text-sm font-medium">{file.name}</h3>
                      <p className="text-xs text-muted-foreground">
                        {(file.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <Button
              className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600"
              onClick={simulateScan}
              disabled={!file || scanning}
            >
              {scanning ? 'Analyzing...' : 'Start Analysis'}
            </Button>

            {scanning && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-3"
              >
                <Progress value={progress} className="h-2 bg-secondary" />
                <div className="flex items-center justify-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent" />
                  <p className="text-sm text-center text-muted-foreground">
                    Analyzing file... {progress}%
                  </p>
                </div>
              </motion.div>
            )}
          </div>
        </Card>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <Card className={`p-6 border-2 ${
                result.status === 'clean' ? 'border-green-500/50' : 'border-red-500/50'
              }`}>
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    {result.status === 'clean' ? (
                      <CheckCircle className="w-6 h-6 text-green-500" />
                    ) : (
                      <XCircle className="w-6 h-6 text-red-500" />
                    )}
                    <h2 className="text-xl font-semibold">
                      {result.status === 'clean' ? 'File is Safe' : 'Potential Threat Detected'}
                    </h2>
                  </div>
                  
                  <div className="space-y-3">
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Confidence Score</p>
                      <div className="flex items-center space-x-2">
                        <div className="flex-1">
                          <Progress 
                            value={result.confidence} 
                            className={`h-3 ${result.status === 'clean' ? '[&>div]:bg-green-500' : '[&>div]:bg-red-500'}`}
                          />
                        </div>
                        <span className="text-lg font-bold">{result.confidence.toFixed(1)}%</span>
                      </div>
                    </div>
                    
                    <div className="bg-secondary/20 p-4 rounded-lg">
                      <p className="text-sm text-muted-foreground">{result.details}</p>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;