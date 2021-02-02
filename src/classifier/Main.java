package classifier;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;

public class Main {
	private static byte[] labels;
	private static float[][] images;
	private static int[] samples;
	private static int numLbl, numImg, numRow, numCol, sampleSize = 3000;
	private static boolean[][] trueLabels;
	private static float[][] weights, wOutput;
	private static float[] biases, bOutput;
	private static float[] bNudge;
	private static float[][] wNudge;
	private static double[] result, z;
	private static float e, numHelper;
	private static final float learningSpeed = 1;
	
	public static void main(String[] args) throws IOException{
		//load training data
		loadData("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
		
		//create vectorization for labels
		trueLabels = new boolean[10][10];
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				if(i == j)
					trueLabels[i][j] = true;
				else
					trueLabels[i][j] = false;
			}
		}
		
		
		//run the training program a given number of times, with different starting weights
		Random rand = new Random();
		weights = new float[numCol*numRow][10];
		biases = new float[10];
		bNudge = new float[10];
		wNudge = new float[numCol*numRow][10];
		samples = new int[sampleSize];
		result = new double[10];
		float[] cost = new float[10];
		z = new double[10];
		double actSum = 0;
		float wSum;
		int numIterations = 1, combo = 0;
		for(int iteration = 0; iteration < numIterations; iteration++) {
		
			//generate initial weights, biases
			for(int i = 0; i < weights.length; i++) {
				for(int j = 0; j < 10; j++) {
					weights[i][j] = (float) ((rand.nextFloat() * 0.999) + 0.001);
				}
			}
			for(float bias : biases) {
				bias = 0;
			}
			
			//do Until minima reached
			int count = 0;
			int totalCost = 0;
			int lastCost = 8000;
			do{
				lastCost = totalCost;
				//reset total cost
				totalCost = 0;
				//generate sampling
				for(float sample : samples) {
					sample = rand.nextInt(numImg);
					//System.out.println(sample);
				}
				
				//for each sample image
				for(int img = 0; img < samples.length; img++) {
					//reset cost
					actSum = 0f;
					
					for(int j = 0; j < 10; j++) {
						//generate output layer
						wSum = 0;
						for(int k = 0; k < numCol*numRow; k++) {
							wSum += images[samples[img]][k] * weights[k][j];
						}
						result[j] = Math.exp(activation(wSum + biases[j]));
						z[j] = wSum + biases[j];
						//System.out.println(result[j]);
						actSum += result[j];
					}
					
					//calculate cost
					for(int i = 0; i < 10; i++) {
						result[i] = (result[i]/(actSum));
						cost[i] = crossEntropy(result[i], trueLabels[labels[samples[img]]][i]);
						totalCost += cost[i];
					}
					
					for(int j = 0; j < 10; j++) {
						//generate bias nudges
						bNudge[j] += dCdZ(j, trueLabels[labels[samples[img]]][j]) * learningSpeed;
						
						//generate weight nudges
						for(int k = 0; k < numCol*numRow; k++) {
							wNudge[k][j] += (float) (dCdZ(j, trueLabels[labels[samples[img]]][j]) * images[samples[img]][k]) * learningSpeed;
						}
					}
					
					
					
				}
				
				//average nudges across images and apply
				for(int j = 0; j < 10; j++) {
					biases[j] += bNudge[j]/(samples.length*10);
				}
				
				for(int j = 0; j < 10; j++) {
					for(int i = 0; i<numCol*numRow; i++) {
						weights[i][j] += wNudge[i][j]/(samples.length * numCol * numRow * 10);
					}
				}
				
				count++;
				System.out.print("Iteration " + count + ", cost: " + totalCost + "\t\t");
				float temp = 0;
				for(int i = 0; i < 10; i++) {
					System.out.print(result[i] + "\t");
					temp += result[i];
				}
				System.out.println(temp);
				/*
				System.out.print("\t\t\t\t");
				for(int i = 0; i < 10; i++) {
					System.out.print(cost[i] + "\t");
					temp += cost[i];
				}
				System.out.println(temp);
				*/
				if(totalCost == lastCost)
					combo++;
				else
					combo = 0;
			}while(combo < 20);
			
			
			
		}
		
		loadData("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");
		
		int score = 0;
		for(int img = 0; img < numImg; img++) {
			for(int j = 0; j < 10; j++) {
				//generate output layer
				wSum = 0;
				for(int k = 0; k < numCol*numRow; k++) {
					wSum += images[img][k] * weights[k][j];
				}
				result[j] = activation(wSum + biases[j]);
			}
			
			//test predicted result
			double bestAns = 0;
			int guess = 0;
			for(int i = 0; i < 10; i++) {
				if(result[i] > bestAns) {
					guess = i;
					bestAns = result[i];
				}
			}
			if(guess == labels[img]) {
				score++;
			}
			
		}
		
		System.out.println("Final score is: " + score + "/" + numImg);
		
		/*
		for(int i = 0; i < numCol; i++) {
			for(int j = 0; j < numRow; j++) {
				System.out.print(images[0][i*numRow + j] + "\t");
			}
			System.out.println();
		}
		*/
		
	}
	
	public static void loadData(String lblFileName, String imgFileName) throws IOException {
		//open training data
		DataInputStream lblStream = new DataInputStream(new FileInputStream(lblFileName));
		DataInputStream imgStream = new DataInputStream(new FileInputStream(imgFileName));
		
		//consume first int
		lblStream.readInt();
		imgStream.readInt();
		
		//read in metadata
		numLbl = lblStream.readInt();
		numImg = imgStream.readInt();
		System.out.println(numImg);
		numRow = imgStream.readInt();
		numCol = imgStream.readInt();
		
		//exit if label and image count do not match
		if (numLbl != numImg) {
			System.out.println("There are " + numLbl + " labels and " + numImg + " images. These must match. Exiting program.");
			System.exit(0);
		}

		//read labels into byte array
		labels = new byte[numLbl];
		lblStream.read(labels);
		
		//read images into byte array
		int numPixels = numCol * numRow;
		byte[] rawImg = new byte[numLbl * numPixels];
		imgStream.read(rawImg);
		images = new float[numImg][numPixels];
		
		//close input streams
		imgStream.close();
		lblStream.close();
		
		//normalize data to 0-1 (0.0001-0.9999)
		for(int i = 0; i < numImg; i++) {
			for(int j = 0; j < numPixels; j++) {
				images[i][j] = (float)((((float)((rawImg[i*numPixels + j] < 0)?rawImg[i*numPixels + j]+256:rawImg[i*numPixels + j]))* 0.999/255) + 0.001);
			}
		}
	}
	
	public static float activation(double x) {
		return (float) (x<0?0:x);
	}
	
	public static float crossEntropy(double real, boolean expected) {
		if(expected)
			return (float) -Math.log(real);
		return (float) -Math.log(1 - real);
	}
	
	public static float dCdZ(int i, boolean expected) {
		if(z[i] <= 0)
			return 0;
		else {
			if(expected)
				return (float) (result[i] - 1);
			else 
				return (float) (-result[i]);
		}
	}
	
	

}
