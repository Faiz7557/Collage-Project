@extends('layouts.guest')

@section('content')
<!-- Hero Section -->
<div class="hero-section d-flex align-items-center justify-content-center text-center">
    <!-- Overlay removed or made very subtle since we use a card now -->
    
    <div class="container position-relative z-index-2">
        <div class="row justify-content-center">
            <div class="col-xl-9 col-lg-10 col-md-11"> <!-- Reduced width -->
                <div class="glass-hero-card rounded-5 animate-up position-relative overflow-hidden" style="padding: 3rem 4rem !important;">
                    <div class="row align-items-center g-5">
                        <!-- Left Column: Content & CTA -->
                        <div class="col-lg-7 text-start">
                            <span class="badge bg-rice-green text-white rounded-pill px-3 py-2 mb-3 fw-bold shadow-sm" style="font-size: 0.8rem; letter-spacing: 1.5px;">
                                <i class="bi bi-stars me-2 text-warning"></i> ECOSYSTEM WARUNG PADI
                            </span>
                            <h1 class="display-5 fw-bold text-dark mb-3 hero-title" style="line-height: 1.2;">
                                Jual Panen Lebih <span class="text-rice-green">Mudah</span><br>
                                Raih Keuntungan <span class="text-rice-gold">Maksimal</span>
                            </h1>
                            <p class="lead text-secondary mb-4 mx-0" style="max-width: 600px; font-weight: 400; font-size: 1.1rem; line-height: 1.6;">
                                Hubungkan hasil panen Anda langsung dengan pembeli potensial. Transaksi adil, transparan, dan menguntungkan.
                            </p>
                            
                            <div class="d-flex justify-content-start gap-3 mb-2">
                                <a href="{{ route('login') }}" class="btn btn-rice-green rounded-pill px-4 py-3 fw-bold shadow-lg hover-scale" style="min-width: 160px;">
                                    Log In
                                </a>
                                <a href="{{ route('register') }}" class="btn btn-outline-success rounded-pill px-4 py-3 fw-bold hover-scale" style="min-width: 160px; border-width: 2px;">
                                    Sign Up
                                </a>
                            </div>
                        </div>

                        <!-- Right Column: Features & Stats Compact Grid -->
                        <div class="col-lg-5">
                            <div class="d-flex flex-column gap-3">
                                <!-- Market Update Bar -->
                                <div class="p-3 rounded-4 glass-sub-card shadow-sm d-flex align-items-center justify-content-between">
                                    <div class="d-flex align-items-center">
                                        <div class="icon-circle-sm bg-warning text-white me-3 shadow-sm"><i class="bi bi-lightning-fill"></i></div>
                                        <div>
                                            <small class="text-uppercase fw-bold text-secondary text-xs d-block mb-1">Gabah Kering</small>
                                            <div class="fw-bold text-dark lh-1">Rp 7.200 <span class="text-success text-xs"><i class="bi bi-caret-up-fill"></i></span></div>
                                        </div>
                                    </div>
                                    <div class="vr"></div>
                                    <div class="text-end ps-2">
                                        <small class="text-uppercase fw-bold text-secondary text-xs d-block mb-1">Petani Active</small>
                                        <div class="fw-bold text-dark lh-1">12,540+</div>
                                    </div>
                                </div>

                                <!-- Features List -->
                                <div class="row g-2">
                                    <div class="col-6">
                                        <div class="p-3 rounded-4 glass-sub-card h-100 text-center hover-glass-dark transition-all">
                                            <i class="bi bi-shield-check fs-3 text-rice-green mb-2 d-block"></i>
                                            <h6 class="fw-bold text-dark mb-0 text-sm">Terverifikasi</h6>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="p-3 rounded-4 glass-sub-card h-100 text-center hover-glass-dark transition-all">
                                            <i class="bi bi-graph-up-arrow fs-3 text-rice-gold mb-2 d-block"></i>
                                            <h6 class="fw-bold text-dark mb-0 text-sm">Transparan</h6>
                                        </div>
                                    </div>
                                    <div class="col-12">
                                        <div class="p-3 rounded-4 glass-sub-card h-100 d-flex align-items-center justify-content-center gap-3 hover-glass-dark transition-all">
                                            <i class="bi bi-truck fs-3 text-info"></i>
                                            <h6 class="fw-bold text-dark mb-0 text-sm">Logistik Cepat & Aman</h6>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Scroll Down Indicator (Optional, can remove if too cluttered) -->
    <a href="#about" class="scroll-down-btn text-dark text-decoration-none d-none d-md-block">
        <span class="small mb-1 d-block text-uppercase fw-bold text-secondary letter-spacing-2">Explore</span>
        <i class="bi bi-arrow-down-circle fs-3 text-rice-green animate-bounce"></i>
    </a>
</div>

<!-- Info / Stats Section (Optional styling match) -->
<div id="about" class="py-5 bg-white position-relative" style="z-index: 2;">
    <div class="container py-5">
        <div class="row g-5 align-items-center">
            <div class="col-md-6 order-md-2">
                 <div class="p-5 rounded-5 bg-light text-center border dashed-border position-relative overflow-hidden">
                    <div class="blob-bg"></div>
                    <img src="{{ asset('images/logo.png') }}" class="img-fluid mb-3 position-relative" style="max-height: 120px;" alt="Logo warung padi">
                    <h5 class="text-secondary position-relative">Solusi Pertanian Masa Depan</h5>
                 </div>
            </div>
            <div class="col-md-6 order-md-1">
                <h2 class="display-6 fw-bold text-dark mb-4" style="font-family: 'Playfair Display', serif;">Memberdayakan Petani<br><span class="text-rice-green">Melalui Inovasi Digital</span></h2>
                <div class="d-flex mb-4">
                    <div class="me-4 text-center">
                        <h2 class="fw-bold text-dark mb-0">10k+</h2>
                        <small class="text-secondary text-uppercase fw-bold" style="font-size: 0.7rem;">Petani</small>
                    </div>
                    <div class="me-4 text-center">
                        <h2 class="fw-bold text-dark mb-0">500+</h2>
                        <small class="text-secondary text-uppercase fw-bold" style="font-size: 0.7rem;">Mitra</small>
                    </div>
                    <div class="text-center">
                        <h2 class="fw-bold text-dark mb-0">98%</h2>
                        <small class="text-secondary text-uppercase fw-bold" style="font-size: 0.7rem;">Kepuasan</small>
                    </div>
                </div>
                <p class="text-muted mb-4" style="line-height: 1.8;">
                    Warung Padi hadir untuk memutus rantai distribusi yang panjang, memberikan harga yang lebih adil bagi petani, dan kualitas terbaik bagi pembeli. Bergabunglah dengan revolusi pertanian digital hari ini.
                </p>
                <a href="{{ route('about') }}" class="btn btn-link text-rice-green fw-bold text-decoration-none p-0">
                    Pelajari Lebih Lanjut <i class="bi bi-arrow-right ms-1"></i>
                </a>
            </div>
        </div>
    </div>
</div>

<style>
    :root {
        --rice-green: #4CAF50;
        --rice-gold: #FF9800;
    }

    /* Brighter, Clearer Glass Card Styles */
    .glass-hero-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 2px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.12);
    }
    
    .icon-circle-sm {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .glass-sub-card {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
    }
    
    .text-xs { font-size: 0.7rem; letter-spacing: 0.5px; }
    .text-sm { font-size: 0.9rem; }
    .lh-1 { line-height: 1; }

    /* Hero Section Styles */
    .hero-section {
        min-height: 100vh;
        background-image: url('{{ asset('images/Background.png') }}');
        background-size: cover;
        background-position: center bottom; /* Anchor image to bottom/center */
        background-repeat: no-repeat;
        position: relative;
        padding-top: 80px; 
        padding-bottom: 40px;
    }
    
    .text-rice-green { color: var(--rice-green) !important; }
    .text-rice-gold { color: var(--rice-gold) !important; }
    .bg-rice-green { background-color: var(--rice-green) !important; }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-weight: 400;
    }
    
    .btn-rice-green {
        background: linear-gradient(135deg, #4CAF50, #66BB6A);
        color: white;
        border: none;
    }
    
    .hover-scale {
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .hover-scale:hover {
        transform: scale(1.05);
    }
    
    /* Scroll Down Indicator */
    .scroll-down-btn {
        position: absolute;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 10;
        text-align: center;
        opacity: 0.8;
        transition: opacity 0.3s;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-10px);}
        60% {transform: translateY(-5px);}
    }
    
    .animate-bounce {
        animation: bounce 2s infinite;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
    .letter-spacing-2 { letter-spacing: 2px; }
    .dashed-border { border: 2px dashed #dee2e6 !important; }
    
    .blob-bg {
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(76, 175, 80, 0.1) 0%, rgba(255,255,255,0) 70%);
        z-index: 0;
    }
</style>
@endsection