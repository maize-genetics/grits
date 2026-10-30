import biokotlin.genome.Position
import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.required
import com.google.common.collect.Range
import com.google.common.collect.RangeMap
import com.google.common.collect.TreeRangeMap
import htsjdk.variant.variantcontext.Allele
import htsjdk.variant.variantcontext.Genotype
import htsjdk.variant.variantcontext.GenotypeBuilder
import htsjdk.variant.variantcontext.GenotypesContext
import htsjdk.variant.variantcontext.VariantContext
import htsjdk.variant.variantcontext.VariantContextBuilder
import htsjdk.variant.variantcontext.writer.Options
import htsjdk.variant.variantcontext.writer.VariantContextWriterBuilder
import htsjdk.variant.vcf.VCFFileReader
import htsjdk.variant.vcf.VCFFormatHeaderLine
import htsjdk.variant.vcf.VCFHeader
import htsjdk.variant.vcf.VCFHeaderLine
import htsjdk.variant.vcf.VCFHeaderLineCount
import htsjdk.variant.vcf.VCFHeaderLineType
import htsjdk.variant.vcf.VCFInfoHeaderLine
import java.io.File
import java.util.HashSet

class BedToVcf : CliktCommand(help = "Convert a set of imputed BED files to a single VCF file") {

    val bedDir: String by option(help="BED directory").required()
    val referencePanelVcf : String by option(help="Reference panel VCF file").required()
    val outFile: String by option(help="Output file").required()


    override fun run() {
        processBedFiles(bedDir, referencePanelVcf, outFile)
    }

    /**
     * Function to run the full bedToVCF process
     */
    fun processBedFiles(bedDir: String, referencePanelVcf: String, outFile: String) {
        val rangeMaps = buildRangeMaps(bedDir)

        buildNewVcf(referencePanelVcf, rangeMaps, outFile)
    }

    /**
     * Function to build a set of range maps from a directory of bed files
     * Each bed file is assumed to be named <sampleId>.bed or <sampleId>_chrX_imputed.bed
     * The range map will map Position ranges to genotype(sampleId) pairs
     * If only one genotype is present in the bed file, it will be used for both genotypes in the pair
     */
    fun buildRangeMaps(bedDir: String): Map<String, RangeMap<Position,Pair<String,String>>> {
        val ranges = mutableMapOf<String, RangeMap<Position,Pair<String,String>>>()

        val bedPattern = Regex(
            """^(.+?)(?:_chr([A-Za-z0-9]+)_imputed)?$"""
        )

        File(bedDir)
            .listFiles { file ->
                file.extension == "bed" &&
                        bedPattern.matches(file.nameWithoutExtension)
            }
            ?.forEach { bedFile ->
                val match = bedPattern.matchEntire(bedFile.nameWithoutExtension)!!

                val sampleId = match.groupValues[1]

                // do work here
                val currentBedRangeMap = createRangeMapFromBedFile(bedFile)
                //check to see if we already have a range map for this sample
                if(ranges.containsKey(sampleId)) {
                    ranges[sampleId]!!.putAll(currentBedRangeMap)
                }
                else {
                    ranges[sampleId] = currentBedRangeMap
                }
            }
        return ranges
    }

    /**
     * Function to create a RangeMap from a BED file
     * The RangeMap will map Position ranges to genotype(sampleId) pairs
     * If only one genotype is present in the bed file, it will be used for both
     * genotypes in the pair
     */
    fun createRangeMapFromBedFile(bedFile: File): RangeMap<Position, Pair<String, String>> {
        // Implement logic to read BED file and create RangeMap
        val rangeMap = TreeRangeMap.create<Position, Pair<String, String>>()
        //Loop through the BED file
        bedFile.forEachLine { line ->
            if(line.startsWith("chrom\tstart\tend\tparent1")) {
                //Skip header lines
                return@forEachLine
            }
            val parts = line.split("\t")
            if (parts.size >= 4) {
                val chrom = parts[0]
                val start = parts[1].toInt()
                val end = parts[2].toInt()
                val genotype1 = parts[3]
                val genotype2 = if (parts.size >=5) parts[4] else genotype1
                val posStart = Position(chrom, start+1 ) // BED is 0-based, Position is 1-based
                val posEnd = Position(chrom, end)
                rangeMap.put(Range.closed(posStart, posEnd), Pair(genotype1, genotype2))
            }
        }

        return rangeMap
    }

    /**
     * Function to build a new VCF file from a reference panel VCF and a set of RangeMaps
     * The RangeMaps map Position ranges to genotype(sampleId) pairs
     * The new VCF will contain genotypes for each imputed sample based on the RangeMaps
     * and the alleles present in the reference panel VCF
     *
     */
    fun buildNewVcf(referencePanelVcf: String, rangeMaps: Map<String, RangeMap<Position,Pair<String,String>>>, outFile: String) {
        // Implement logic to read reference VCF and write new VCF using rangeMaps
        // This is a placeholder for the actual implementation
        println("Building new VCF from $referencePanelVcf using BED data and writing to $outFile")
        //Build a VCF writer
        VariantContextWriterBuilder()
            .unsetOption(Options.INDEX_ON_THE_FLY)
            .setOutputFile(File(outFile))
            .setOutputFileType(VariantContextWriterBuilder.OutputType.VCF)
            .setOption(Options.ALLOW_MISSING_FIELDS_IN_HEADER)
            .build().use { writer ->
                writer.writeHeader(createGenericHeader(rangeMaps.keys.toList(), emptySet()))


                //Open up the reference VCF and loop through each record
                val variantReader = VCFFileReader(File(referencePanelVcf), false)

                val iterator = variantReader.iterator()

                while(iterator.hasNext()) {
                    val vc = iterator.next()

                    val gameteToAlleleMap = buildGameteToAlleleMap(vc)

                    val newVc = buildNewVariantContext(rangeMaps, vc, gameteToAlleleMap)

                    //Write the new variant context to the VCF
                    writer.add(newVc)

                }

            }
    }

    /**
     * Function to build a new VariantContext from a reference VariantContext and a set of RangeMaps
     * The RangeMaps map Position ranges to genotype(sampleId) pairs
     * The new VariantContext will contain genotypes for each imputed sample based on the RangeMaps
     * and the alleles present in the reference VariantContext
     */
    fun buildNewVariantContext(
        rangeMaps: Map<String, RangeMap<Position, Pair<String, String>>>,
        vc: VariantContext,
        gameteToAlleleMap: Map<String, Allele>
    ): VariantContext {
        val pos = Position(vc.contig, vc.start)
        val genotypeList = buildNewGenotypes(rangeMaps, pos, gameteToAlleleMap)

        // AC/AN computed from the ORIGINAL reference panel's own genotypes
        // (the 25 founders) at this site -- NOT the newly-built imputed
        // sample's genotype -- so downstream frequency-binned diagnostics
        // (compare_gvcf_truth.py / vcf_eval/accuracy.py's
        // update_frequency_bins(), which reads AC/AN straight from this
        // INFO field) can bin by how common/rare this allele actually is
        // across the reference panel. `referencePanelVcf` (merge-gvcfs
        // output) never annotates AC/AN itself, and VariantContextBuilder(vc)
        // only copies attributes that already exist on vc, so this was
        // always "." -- silently zeroing out every AF-binned accuracy/R2
        // table this pipeline has ever produced (both in-panel and
        // held-out reports).
        var an = 0
        var ac = 0
        for (genotype in vc.genotypes) {
            for (allele in genotype.alleles) {
                if (allele.isCalled) {
                    an++
                    if (allele.isNonReference) {
                        ac++
                    }
                }
            }
        }

        //Build a new variant context with the new genotype
        val newVc = VariantContextBuilder(vc)
            .genotypes(genotypeList)
            .attribute("AC", ac)
            .attribute("AN", an)
            .make()
        return newVc
    }

    /**
     * Function to build a list of new Genotypes from a set of RangeMaps and a Position
     * This will lookup the postion in each RangeMap to get the genotype(sampleId) pair
     * and then use the gameteToAlleleMap to get the corresponding Alleles
     * to build a new Genotype list.
     */
    fun buildNewGenotypes(
        rangeMaps: Map<String, RangeMap<Position, Pair<String, String>>>,
        pos: Position,
        gameteToAlleleMap: Map<String, Allele>
    ): MutableList<Genotype> {
        val genotypeList = mutableListOf<Genotype>()

        for (sampleName in rangeMaps.keys) {
            val rangeMap = rangeMaps[sampleName]!!
            val genotypePair = rangeMap.get(pos)
            if (genotypePair != null) {
                val gamete1 = genotypePair.first
                val gamete2 = genotypePair.second

                val allele1 = gameteToAlleleMap[gamete1] ?: Allele.NO_CALL
                val allele2 = gameteToAlleleMap[gamete2] ?: Allele.NO_CALL
                //Build a new genotype object
                val newGenotype = GenotypeBuilder(sampleName)
                    .alleles(listOf(allele1, allele2))
                    .make()

                genotypeList.add(newGenotype)
            }
        }
        return genotypeList
    }

    /**
     * Function to create a generic VCF header with standard lines and additional lines
     * This is needed by the VCF writer
     */
    fun createGenericHeader(taxaNames: List<String>, altLines:Set<VCFHeaderLine>): VCFHeader {
        val headerLines = createGenericHeaderLineSet() as MutableSet<VCFHeaderLine>
        headerLines.addAll(altLines)
        return VCFHeader(headerLines, taxaNames)
    }

    /**
     * Function to create a generic set of VCF header lines
     */
    fun createGenericHeaderLineSet(): Set<VCFHeaderLine> {
        val headerLines: MutableSet<VCFHeaderLine> = HashSet()
        headerLines.add(VCFFormatHeaderLine("AD", 3, VCFHeaderLineType.Integer, "Allelic depths for the ref and alt alleles in the order listed"))
        headerLines.add(
            VCFFormatHeaderLine("DP", 1, VCFHeaderLineType.Integer, "Read Depth (only filtered reads used for calling)")
        )
        headerLines.add(VCFFormatHeaderLine("GQ", 1, VCFHeaderLineType.Integer, "Genotype Quality"))
        headerLines.add(VCFFormatHeaderLine("GT", 1, VCFHeaderLineType.String, "Genotype"))
        headerLines.add(
            VCFFormatHeaderLine("PL", VCFHeaderLineCount.G, VCFHeaderLineType.Integer, "Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification")
        )
        headerLines.add(VCFInfoHeaderLine("DP", 1, VCFHeaderLineType.Integer, "Total Depth"))
        headerLines.add(VCFInfoHeaderLine("NS", 1, VCFHeaderLineType.Integer, "Number of Samples With Data"))
        headerLines.add(VCFInfoHeaderLine("AF", 3, VCFHeaderLineType.Integer, "Allele Frequency"))
        headerLines.add(VCFInfoHeaderLine("END", 1, VCFHeaderLineType.Integer, "Stop position of the interval"))
        // These last 4 are needed for assembly g/hvcfs, but not for reference.  I am keeping them in as header
        // lines but they will only be added to the data lines if they are present in the VariantContext.
        headerLines.add(VCFInfoHeaderLine("ASM_Chr", 1, VCFHeaderLineType.String, "Assembly chromosome"))
        headerLines.add(VCFInfoHeaderLine("ASM_Start", 1, VCFHeaderLineType.Integer, "Assembly start position"))
        headerLines.add(VCFInfoHeaderLine("ASM_End", 1, VCFHeaderLineType.Integer, "Assembly end position"))
        headerLines.add(VCFInfoHeaderLine("ASM_Strand", 1, VCFHeaderLineType.String, "Assembly strand"))
        return headerLines
    }

    /**
     * Function to build a map of gamete names to Alleles from a VariantContext
     * This assumes that the sample names in the VariantContext correspond to gamete names
     * and that each sample has a single allele representing the gamete's allele at this position
     */
    fun buildGameteToAlleleMap(vc: VariantContext): Map<String, Allele> {
        //Loop through each sample in the variant context
        return vc.genotypes.map { genotype ->
            val sampleName = genotype.sampleName
            val alleles = genotype.alleles
            if(alleles.isNotEmpty()) {
                Pair(sampleName, alleles[0])
            }
            else {
                Pair("", Allele.NO_CALL)
            }
        }.filter { it.first != "" }.toMap()
    }
}