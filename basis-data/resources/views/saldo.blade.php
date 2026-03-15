@extends('layouts.main')

@section('content')
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card saldo-card animate-fade-in">
                <!-- Custom Header dengan Gradient Orange -->
                <div class="card-header-custom d-flex align-items-center justify-content-center">
                    <i class="fas fa-coins me-3 text-white"></i>
                    <h2 class="text-white mb-0 font-weight-bold">Saldo Anda</h2>
                </div>

                <div class="card-body">
                    <!-- Display Saldo Utama -->
                    <div class="saldo-display text-center mb-4">
                        <i class="fas fa-wallet fs-1 text-success mb-3"></i>
                        <h1 class="text-4xl font-bold text-white mb-2">Rp {{ number_format($saldo, 0, ',', '.') }}</h1>
                        <p class="text-light">Saldo saat ini tersedia untuk transaksi</p>
                    </div>

                    @if(session('success'))
                        <div class="alert alert-success alert-custom mb-4 animate-slide-in">
                            <i class="fas fa-check-circle me-2"></i>
                            {{ session('success') }}
                        </div>
                    @endif

                    @if(session('error'))
                        <div class="alert alert-danger alert-custom mb-4 animate-slide-in">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            {{ session('error') }}
                        </div>
                    @endif

                    @if($pendingTopUp)
                        <!-- Pending Top-Up Section -->
                        <div class="pending-section mb-4">
                            <div class="alert alert-info alert-custom mb-3">
                                <i class="fas fa-clock me-2"></i>
                                <p><strong>Permintaan Top-up Belum Dikonfirmasi</strong></p>
                                <small class="text-light">Pastikan pembayaran telah dilakukan menggunakan kode referensi di bawah.</small>
                            </div>

                            <div class="row">
                                <!-- Kode Referensi -->
                                <div class="col-md-6 mb-3">
                                    <div class="info-box ref-code">
                                        <h5 class="text-white mb-2"><i class="fas fa-barcode me-2"></i>Kode Referensi</h5>
                                        <div class="code-display bg-light text-center p-3 rounded">
                                            <h4 class="mb-0 font-monospace">{{ $pendingTopUp->reference_code }}</h4>
                                        </div>
                                        <small class="text-light d-block mt-1">Gunakan kode ini saat pembayaran</small>
                                    </div>
                                </div>
                                
                                <!-- Jumlah -->
                                <div class="col-md-6 mb-3">
                                    <div class="info-box amount-box">
                                        <h5 class="text-white mb-2"><i class="fas fa-money-bill-wave me-2"></i>Jumlah</h5>
                                        <h4 class="text-success mb-0">Rp {{ number_format($pendingTopUp->amount, 0, ',', '.') }}</h4>
                                        <small class="text-light">Jumlah yang diminta</small>
                                    </div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <h5 class="text-white mb-2"><i class="fas fa-credit-card me-2"></i>Metode Pembayaran</h5>
                                <div class="payment-method bg-light p-3 rounded text-center">
                                    <p class="lead mb-0">{{ $pendingTopUp->payment_method == 'bank' ? 'Bank Transfer' : 'Mini Market' }}</p>
                                </div>
                            </div>

                            <form action="{{ route('saldo.topup.confirm', $pendingTopUp->id) }}" method="POST" class="text-center">
                                @csrf
                                <button type="submit" class="btn btn-success btn-lg px-4">
                                    <i class="fas fa-check me-2"></i>Sudah Melakukan Pembayaran
                                </button>
                            </form>
                        </div>
                    @else
                        <!-- Top-Up Form -->
                        <div class="topup-section">
                            <h5 class="text-white mb-4 text-center">
                                <i class="fas fa-plus-circle me-2"></i>Top-up Saldo
                            </h5>
                            <form method="POST" action="{{ route('saldo.topup.store') }}" class="needs-validation" novalidate>
                                @csrf

                                <div class="form-group mb-4">
                                    <label for="amount" class="form-label text-white">Jumlah Top-up (Rp)</label>
                                    <div class="input-group">
                                        <span class="input-group-text bg-transparent border-light text-white">Rp</span>
                                        <input id="amount" type="number" class="form-control form-input-custom @error('amount') is-invalid @enderror" name="amount" value="{{ old('amount') }}" required autofocus min="10000" placeholder="Masukkan jumlah minimal Rp 10.000">
                                        @error('amount')
                                            <span class="invalid-feedback" role="alert">
                                                <strong>{{ $message }}</strong>
                                            </span>
                                        @enderror
                                    </div>
                                    <small class="text-light">Minimal Rp 10.000</small>
                                </div>

                                <div class="form-group mb-4">
                                    <label for="payment_method" class="form-label text-white">Metode Pembayaran</label>
                                    <select id="payment_method" class="form-control form-select-custom @error('payment_method') is-invalid @enderror" name="payment_method" required>
                                        <option value="" disabled selected>Pilih metode</option>
                                        <option value="bank" {{ old('payment_method') == 'bank' ? 'selected' : '' }}>Bank Transfer</option>
                                        <option value="mini_market" {{ old('payment_method') == 'mini_market' ? 'selected' : '' }}>Mini Market</option>
                                    </select>
                                    @error('payment_method')
                                        <span class="invalid-feedback" role="alert">
                                            <strong>{{ $message }}</strong>
                                        </span>
                                    @enderror
                                </div>

                                <div class="form-group text-center">
                                    <button type="submit" class="btn btn-primary btn-lg px-5">
                                        <i class="fas fa-arrow-right me-2"></i>Lanjutkan Top-up
                                    </button>
                                </div>
                            </form>
                        </div>
                    @endif
                </div>
            </div>
        </div>
    </div>
</div>

<style>
    .saldo-card {
        background: linear-gradient(135deg, #4CAF50, #81C784);
        backdrop-filter: blur(10px);
        border: 1px solid #FF9800;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: #fff;
    }
    
    .saldo-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FF9800, #4CAF50);
    }
    
    .saldo-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    
    .card-header-custom {
        background: linear-gradient(135deg, #FF9800, #FFB74D);
        border: none;
        border-radius: 16px 16px 0 0;
        padding: 20px;
        color: #fff;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .saldo-display h1 {
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .alert-custom {
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        border-left: 4px solid;
        padding-left: 20px;
        color: #fff;
    }
    
    .alert-success { background: rgba(40, 167, 69, 0.2); border-left-color: #28a745; }
    .alert-danger { background: rgba(220, 53, 69, 0.2); border-left-color: #dc3545; }
    .alert-info { background: rgba(23, 162, 184, 0.2); border-left-color: #17a2b8; }
    
    .pending-section, .topup-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }
    
    .code-display, .payment-method {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #333;
        font-weight: 600;
    }
    
    .form-input-custom, .form-select-custom {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 8px;
        color: #fff;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .form-input-custom::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
    
    .form-input-custom:focus, .form-select-custom:focus {
        background: rgba(255, 255, 255, 0.3);
        border-color: #FF9800;
        box-shadow: 0 0 0 0.2rem rgba(255, 152, 0, 0.25);
        color: #fff;
    }
    
    .form-input-custom::-webkit-input-placeholder { color: rgba(255, 255, 255, 0.7); }
    .form-input-custom::-moz-placeholder { color: rgba(255, 255, 255, 0.7); }
    
    .input-group-text {
        background: rgba(255, 152, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.5);
        color: #fff;
    }
    
    .btn {
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .btn-success { background: linear-gradient(135deg, #28a745, #20c997); border: none; }
    .btn-primary { background: linear-gradient(135deg, #007bff, #0056b3); border: none; }
    
    .btn-success:hover, .btn-primary:hover { opacity: 0.9; }
    
    .text-light { color: rgba(255, 255, 255, 0.8) !important; }
    
    /* Animasi */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .animate-fade-in { animation: fadeInUp 0.6s ease-out forwards; }
    .animate-slide-in { animation: slideIn 0.4s ease-out forwards; }
    
    /* Responsiveness */
    @media (max-width: 768px) {
        .saldo-card { margin: 1rem; }
        .text-4xl { font-size: 2.5rem !important; }
        .card-header-custom { padding: 15px; flex-direction: column; text-align: center; }
        .pending-section, .topup-section { padding: 15px; }
    }
</style>
@endsection