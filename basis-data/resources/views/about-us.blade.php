```blade
@extends('layouts.guest')

@section('content')
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    .about-us-hero {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 50%, #2E7D32 100%);
        color: #fff;
        padding: 80px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 0;
    }
    .about-us-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
        pointer-events: none;
    }
    .about-us-hero h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        animation: fadeInUp 1s ease-out;
    }
    .about-us-hero .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 40px;
        position: relative;
        z-index: 1;
        opacity: 0.9;
        animation: fadeInUp 1s ease-out 0.2s both;
    }

    .about-us-container {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #333;
        padding: 60px 20px;
        min-height: 100vh;
        position: relative;
    }
    .about-us-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #4CAF50, transparent);
    }

    .story-section {
        max-width: 1200px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: 1fr;
        gap: 50px;
    }
    .story-card {
        background: #fff;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(76, 175, 80, 0.1);
    }
    .story-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(to right, #4CAF50, #FFA726);
    }
    .story-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    .story-card h2.section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 25px;
        color: #4CAF50;
        position: relative;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .story-card h2.section-title::after {
        content: '';
        width: 50px;
        height: 3px;
        background: linear-gradient(to right, #4CAF50, #FFA726);
        border-radius: 2px;
    }
    .story-card p {
        font-size: 1.15rem;
        line-height: 1.9;
        margin-bottom: 25px;
        text-align: justify;
        color: #555;
        animation: fadeIn 1s ease-out;
    }
    .story-card .highlight-orange {
        background: linear-gradient(to right, #FFA726, #FF8F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: #FF8F00; /* Fallback */
        font-weight: 600;
        position: relative;
    }
    .story-card .highlight-orange:hover {
        text-shadow: 0 0 10px rgba(255, 167, 38, 0.5);
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .story-card:nth-child(1) { animation-delay: 0.1s; }
    .story-card:nth-child(2) { animation-delay: 0.2s; }
    .story-card:nth-child(3) { animation-delay: 0.3s; }

    /* Responsive Design */
    @media (max-width: 768px) {
        .about-us-hero {
            padding: 60px 15px;
        }
        .about-us-hero h1 {
            font-size: 2.5rem;
        }
        .about-us-hero .subtitle {
            font-size: 1.1rem;
        }
        .about-us-container {
            padding: 40px 15px;
        }
        .story-card {
            padding: 30px 20px;
            margin-bottom: 30px;
        }
        .story-card h2.section-title {
            font-size: 1.8rem;
        }
        .story-card p {
            font-size: 1rem;
        }
    }
    @media (min-width: 769px) and (max-width: 1024px) {
        .story-section {
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
        }
    }
</style>

<div class="container-fluid p-0">
    <!-- Hero Section -->
    <section class="about-us-hero">
        <div style="position: relative; z-index: 1;">
            <h1>Tentang Kami</h1>
            <p class="subtitle">Membangun Ekosistem Beras yang Transparan dan Adil untuk Indonesia</p>
        </div>
    </section>

    <!-- Main Content -->
    <section class="about-us-container">
        <div class="story-section">
            <div class="story-card">
                <h2 class="section-title">
                    <span>🌾</span> Cerita Kami
                </h2>
                <p>
                    Sejak awal, kami melihat <span class="highlight-orange">tantangan besar</span> dalam rantai pasokan beras di Indonesia. Petani sering
                    kali <span class="highlight-orange">kesulitan</span> menjual hasil panennya dengan <span class="highlight-orange">harga yang adil</span>, sementara pihak pengepul dan
                    distributor kesulitan mendapatkan pasokan yang <span class="highlight-orange">konsisten dan berkualitas</span>. Rantai distribusi
                    yang <span class="highlight-orange">panjang dan rumit</span> sering kali tidak transparan, menyebabkan ketidakadilan dan inefisiensi
                    di seluruh ekosistem.
                </p>
            </div>

            <div class="story-card">
                <h2 class="section-title">
                    <span>🚀</span> Solusi Kami
                </h2>
                <p>
                    Melihat masalah ini, <span class="highlight-orange">kami hadir</span> untuk menjadi <span class="highlight-orange">solusi</span>. Kami membangun platform digital yang
                    menghubungkan <span class="highlight-orange">semua pihak</span> secara langsung, mulai dari petani, pengepul, hingga distributor.
                    Tujuannya sederhana: menciptakan ekosistem yang <span class="highlight-orange">transparan dan efisien</span>, di mana setiap
                    orang dapat berinteraksi, bernegosiasi, dan bertransaksi <span class="highlight-orange">secara adil</span>. Kami percaya bahwa
                    dengan teknologi, kami bisa memotong mata rantai yang tidak perlu, sehingga petani bisa
                    mendapatkan <span class="highlight-orange">keuntungan lebih</span> dan pembeli bisa memperoleh <span class="highlight-orange">beras terbaik</span> langsung dari
                    sumbernya.
                </p>
            </div>

            <div class="story-card">
                <h2 class="section-title">
                    <span>🤝</span> Komunitas Kami
                </h2>
                <p>
                    Kami tidak hanya membangun platform, tetapi juga <span class="highlight-orange">sebuah komunitas</span>. Komunitas yang
                    didasari pada kepercayaan, di mana setiap transaksi adalah langkah maju menuju <span class="highlight-orange">ketahanan
                    pangan</span> yang lebih baik bagi seluruh negeri. Melalui visi dan komitmen ini, kami berharap dapat
                    membawa <span class="highlight-orange">perubahan positif</span> dan <span class="highlight-orange">berkelanjutan</span> dalam industri beras Indonesia.
                </p>
            </div>
        </div>
    </section>
</div>
@endsection
```