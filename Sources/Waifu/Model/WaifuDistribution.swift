import Foundation

/// Whether this build is licensed to offer every model, or only the ones whose
/// upstream licence permits commercial use.
///
/// ## Why this exists
///
/// WaifuFinder's own code is MIT, which permits commercial use, sublicensing and
/// resale. Most of what it drives is permissively licensed too: SeedVR2, FlashVSR,
/// BasicSR and the Real-ESRGAN weights are Apache-2.0 or BSD-3-Clause, and the MLX,
/// mlx-swift and InferKit code it links is MIT. Those combine freely.
///
/// **PGTFormer does not.** Its upstream licence permits redistribution and use "for
/// non-commercial purposes" only, and commercial use requires permission from Xidian
/// University or the contributors. A non-commercial term cannot be satisfied by an
/// MIT licence, and the two cannot be reconciled by attribution or by the code being
/// separable — what matters is the *use*, not how the files are packaged.
///
/// The exposure is not hypothetical. The app fetches models anonymously at the user's
/// request, so a commercial user who taps Download for PGTFormer is infringing, and an
/// MIT-licensed app that ships that button is arguably contributing to it. Offering
/// the model and refusing at runtime would be worse: the offer is the problem.
///
/// So the choice is made at build time instead. A non-commercial build offers
/// PGTFormer. A build made with `WAIFU_COMMERCIAL_DISTRIBUTION=1` does not offer it at
/// all — not in Settings, not in the algorithm selector, and not as a downloadable
/// release — leaving an MIT/Apache/BSD product that is clean for commercial use.
///
/// ## What this does not decide
///
/// Two questions need a human, not a build flag:
///
/// - **The weights carry no licence of their own.** `kepeng/pgtformer-base` on the
///   Hugging Face Hub declares no licence in its card metadata and ships no LICENSE
///   file. The only terms available are the non-commercial LICENSE in the author's
///   GitHub repository. A reasonable reading is that the author intended his
///   repository terms to cover the weights he published, but that is an inference,
///   not a grant. Absent an explicit grant, default copyright applies and nobody —
///   including us — has a documented right to redistribute them. This is worth
///   asking the author about before relying on it.
/// - **Which distribution is actually commercial.** A build-time flag records an
///   intention; it does not test it. Charging money, or using the app in a business
///   process, is what the licence turns on.
///
/// Neither is a question this code should answer on the developer's behalf. Get a
/// lawyer's answer before distributing commercially.
enum WaifuDistribution {
    /// `true` when this build must not be used commercially, because it offers a
    /// model whose licence forbids it.
    ///
    /// Set by compiling with `-D WAIFU_COMMERCIAL_DISTRIBUTION` for a commercial
    /// build; `./build.sh` does this when `WAIFU_COMMERCIAL_DISTRIBUTION=1` is in the
    /// environment. The default build is the non-commercial one.
    static var isNonCommercialOnly: Bool {
        #if WAIFU_COMMERCIAL_DISTRIBUTION
        false
        #else
        true
        #endif
    }
}